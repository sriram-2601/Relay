"""
Calibrated Stand-in Model Service for 'The Relay'.
Calibrated directly from Groq Free Tier (Llama 3.3 70B / Llama 3.1 8B):
- Concurrency queueing and latency degradation under load
- Streaming token chunk emission (piece at a time)
- Rate limiting: RPM (30), TPM (12,000), RPD (1,000), TPD (100,000)
- Emits exact Groq-compatible 429 error structures with retry-after
- Simulates network timeouts, failures, and daily budget exhaustion
"""

import time
import random
import re
import threading
import asyncio
from typing import AsyncGenerator, Generator, Dict, Any, List, Optional, Tuple

class GroqRateLimitError(Exception):
    def __init__(self, message: str, retry_after: float, limit_type: str = "tokens"):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.status_code = 429
        self.error_body = {
            "error": {
                "message": message,
                "type": limit_type,
                "code": "rate_limit_exceeded"
            }
        }

class GroqTimeoutError(Exception):
    def __init__(self, message: str = "Gateway Timeout"):
        super().__init__(message)
        self.status_code = 504

class CalibratedStandInModel:
    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 30000, # Groq Free Tier (Llama 3.1 8B: 30k TPM, 30 RPM)
        rpd_limit: int = 1000,
        tpd_limit: int = 100000,
        base_ttft_ms: float = 180.0,
        ms_per_token: float = 14.0, # Groq ~70-100 tok/sec
        failure_rate: float = 0.02
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit
        
        self.base_ttft_ms = base_ttft_ms
        self.ms_per_token = ms_per_token
        self.failure_rate = failure_rate
        
        # Concurrency tracking
        self.in_flight_requests = 0
        self.peak_in_flight = 0
        
        # Rolling window telemetry: timestamps of requests & tokens
        self.request_timestamps: List[float] = []
        self.token_timestamps: List[Tuple[float, int]] = []
        
        # Cumulative daily counters
        self.daily_requests_count = 0
        self.daily_tokens_count = 0
        self.daily_budget_exhausted = False
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

    def _check_and_update_limits_sync(self, estimated_tokens_in: int, estimated_tokens_out: int):
        with self._thread_lock:
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60.0]
            self.token_timestamps = [item for item in self.token_timestamps if now - item[0] < 60.0]
            
            if self.daily_budget_exhausted or (self.daily_requests_count >= self.rpd_limit) or (self.daily_tokens_count >= self.tpd_limit):
                raise GroqRateLimitError(
                    message="Daily quota exceeded for model 'llama-3.3-70b-versatile'. Your quota will reset at 00:00 UTC.",
                    retry_after=32400.0,
                    limit_type="requests_per_day"
                )
            
            if len(self.request_timestamps) >= self.rpm_limit:
                oldest = self.request_timestamps[0]
                wait_sec = round(max(0.5, 60.0 - (now - oldest)), 1)
                raise GroqRateLimitError(
                    message=f"Rate limit reached for requests per minute (RPM={self.rpm_limit}). Please try again in {wait_sec}s.",
                    retry_after=wait_sec,
                    limit_type="requests"
                )
            
            current_minute_tokens = sum(count for _, count in self.token_timestamps)
            total_req_tokens = estimated_tokens_in + estimated_tokens_out
            if current_minute_tokens + total_req_tokens > self.tpm_limit:
                oldest_token_time = self.token_timestamps[0][0] if self.token_timestamps else now
                wait_sec = round(max(0.5, 60.0 - (now - oldest_token_time)), 1)
                raise GroqRateLimitError(
                    message=f"Rate limit reached for tokens per minute (TPM={self.tpm_limit}). Wait {wait_sec}s.",
                    retry_after=wait_sec,
                    limit_type="tokens"
                )
                
            if random.random() < self.failure_rate:
                if random.random() < 0.5:
                    raise GroqTimeoutError("504 Gateway Timeout: Groq upstream worker timed out")
                else:
                    raise Exception("500 Internal Server Error: Model service busy")

            self.request_timestamps.append(now)
            self.token_timestamps.append((now, total_req_tokens))
            self.daily_requests_count += 1
            self.daily_tokens_count += total_req_tokens

    def stream_chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        prompt_text = " ".join([m.get("content", "") for m in messages])
        tokens_in = max(10, int(len(prompt_text.split()) * 1.3))
        response_text = self._synthesize_grounded_response(prompt_text)
        response_words = response_text.split()
        tokens_out = max(5, int(len(response_words) * 1.3))

        self._check_and_update_limits_sync(tokens_in, tokens_out)

        with self._thread_lock:
            self.in_flight_requests += 1
            if self.in_flight_requests > self.peak_in_flight:
                self.peak_in_flight = self.in_flight_requests
            current_in_flight = self.in_flight_requests

        try:
            queue_delay_ms = 18.0 * (current_in_flight ** 1.8) if current_in_flight > 1 else 0.0
            jitter_ms = random.uniform(-25.0, 45.0)
            prompt_process_ms = min(300.0, tokens_in * 0.08)
            simulated_ttft_ms = max(80.0, self.base_ttft_ms + prompt_process_ms + queue_delay_ms + jitter_ms)
            
            time.sleep(simulated_ttft_ms / 1000.0)

            chunk_size_words = 3
            for i in range(0, len(response_words), chunk_size_words):
                chunk = " ".join(response_words[i:i + chunk_size_words]) + " "
                yield chunk
                chunk_delay_ms = self.ms_per_token * (len(chunk.split()) * 1.3)
                time.sleep(chunk_delay_ms / 1000.0)
        finally:
            with self._thread_lock:
                self.in_flight_requests = max(0, self.in_flight_requests - 1)

    def set_halved_limit(self, enabled: bool = True):
        """Halves per-minute RPM/TPM limits to simulate upstream throttling."""
        if enabled:
            self.rpm_limit = 15
            self.tpm_limit = 15000
        else:
            self.rpm_limit = 30
            self.tpm_limit = 30000

    def set_daily_exhausted(self, exhausted: bool = True):
        """Simulates 3pm daily budget exhaustion."""
        self.daily_budget_exhausted = exhausted

    async def _check_and_update_limits(self, estimated_tokens_in: int, estimated_tokens_out: int):
        async with self._lock:
            now = time.time()
            
            # Purge timestamps older than 60 seconds
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60.0]
            self.token_timestamps = [item for item in self.token_timestamps if now - item[0] < 60.0]
            
            # Check Daily Budget
            if self.daily_budget_exhausted or (self.daily_requests_count >= self.rpd_limit) or (self.daily_tokens_count >= self.tpd_limit):
                raise GroqRateLimitError(
                    message="Daily quota exceeded for model 'llama-3.3-70b-versatile'. Your quota will reset at 00:00 UTC.",
                    retry_after=32400.0, # e.g. 9 hours until midnight UTC
                    limit_type="requests_per_day"
                )
            
            # Check RPM (Requests Per Minute)
            if len(self.request_timestamps) >= self.rpm_limit:
                oldest = self.request_timestamps[0]
                wait_sec = round(max(0.5, 60.0 - (now - oldest)), 1)
                raise GroqRateLimitError(
                    message=f"Rate limit reached for requests per minute (RPM={self.rpm_limit}). Please try again in {wait_sec}s.",
                    retry_after=wait_sec,
                    limit_type="requests"
                )
            
            # Check TPM (Tokens Per Minute)
            current_minute_tokens = sum(count for _, count in self.token_timestamps)
            total_req_tokens = estimated_tokens_in + estimated_tokens_out
            if current_minute_tokens + total_req_tokens > self.tpm_limit:
                oldest_token_time = self.token_timestamps[0][0] if self.token_timestamps else now
                wait_sec = round(max(0.5, 60.0 - (now - oldest_token_time)), 1)
                raise GroqRateLimitError(
                    message=f"Rate limit reached for tokens per minute (TPM={self.tpm_limit}). Current minute used: {current_minute_tokens}. Requested: {total_req_tokens}. Wait {wait_sec}s.",
                    retry_after=wait_sec,
                    limit_type="tokens"
                )
                
            # Check for transient random server failure / gateway timeout
            if random.random() < self.failure_rate:
                # 50% chance of timeout vs 500 error
                if random.random() < 0.5:
                    raise GroqTimeoutError("504 Gateway Timeout: Groq upstream worker timed out")
                else:
                    raise Exception("500 Internal Server Error: Model service busy")

            # Book the request and tokens
            self.request_timestamps.append(now)
            self.token_timestamps.append((now, total_req_tokens))
            self.daily_requests_count += 1
            self.daily_tokens_count += total_req_tokens

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """
        Simulates realistic streaming chat completion with TTFT,
        load degradation, and chunk-by-chunk token emission.
        """
        # Estimate prompt tokens (~1.3 tokens per word)
        prompt_text = " ".join([m.get("content", "") for m in messages])
        tokens_in = max(10, int(len(prompt_text.split()) * 1.3))
        
        # Generate representative response based on content
        response_text = self._synthesize_grounded_response(prompt_text)
        response_words = response_text.split()
        tokens_out = max(5, int(len(response_words) * 1.3))

        # 1. Check rate limits & enforce quotas
        await self._check_and_update_limits(tokens_in, tokens_out)

        # 2. Track in-flight requests & concurrency load degradation
        async with self._lock:
            self.in_flight_requests += 1
            if self.in_flight_requests > self.peak_in_flight:
                self.peak_in_flight = self.in_flight_requests
            current_in_flight = self.in_flight_requests

        try:
            # Calibrated TTFT formula:
            # Base latency (180ms) + prompt length impact + quadratic concurrency degradation under burst
            queue_delay_ms = 18.0 * (current_in_flight ** 1.8) if current_in_flight > 1 else 0.0
            jitter_ms = random.uniform(-25.0, 45.0)
            prompt_process_ms = min(300.0, tokens_in * 0.08)
            
            simulated_ttft_ms = max(80.0, self.base_ttft_ms + prompt_process_ms + queue_delay_ms + jitter_ms)
            
            # Sleep until first token
            await asyncio.sleep(simulated_ttft_ms / 1000.0)

            # 3. Stream tokens piece at a time
            # Group into 2-4 words per streaming chunk like Groq/SSE
            chunk_size_words = 3
            for i in range(0, len(response_words), chunk_size_words):
                chunk = " ".join(response_words[i:i + chunk_size_words]) + " "
                yield chunk
                # Inter-token generation delay
                chunk_delay_ms = self.ms_per_token * (len(chunk.split()) * 1.3)
                await asyncio.sleep(chunk_delay_ms / 1000.0)

        finally:
            async with self._lock:
                self.in_flight_requests = max(0, self.in_flight_requests - 1)

    def _synthesize_grounded_response(self, prompt: str) -> str:
        """
        Produces realistic grounded responses with exact paper citations
        or honest refusal when context is missing.
        """
        p_lower = prompt.lower()
        
        # Check if prompt explicitly conveys unanswerable/out-of-corpus/refusal condition
        if "out-of-corpus" in p_lower or "cannot answer" in p_lower or "unanswerable" in p_lower or "refuse" in p_lower:
            return (
                "Based on the provided library of papers ('Attention Is All You Need', "
                "'Moshi: a speech-text foundation model for real-time dialogue', and "
                "'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling'), "
                "the papers do not contain information to answer this question. "
                "The assistant refuses to guess without explicit textual grounding in the library."
            )

        # Grounded answer templates matching paper content based on specific paper targets
        if "attention is all you need" in p_lower or "d_model" in p_lower or ("transformer" in p_lower and "moshi" not in p_lower):
            return (
                "In the Transformer architecture, the multi-head attention mechanism employs h=8 parallel attention heads. "
                "For each head, the projection dimensions are d_k = d_v = d_model / h = 64, with the total model dimension "
                "d_model = 512 [Attention Is All You Need, Section 3.2.2, Page 5]. The feed-forward inner layer dimension "
                "is d_ff = 2048 [Attention Is All You Need, Section 3.3, Page 5]."
            )
        elif "dynamic chunking" in p_lower or "h-net" in p_lower or "hierarchical sequence" in p_lower:
            return (
                "H-Net utilizes dynamic chunking to replace heuristic BPE tokenizers with an end-to-end learned "
                "hierarchical sequence model [Dynamic Chunking, Section 1, Page 2]. The chunking module computes "
                "boundary transition probabilities using a lightweight predictor over raw byte embeddings, achieving "
                "up to 4x data efficiency gains on non-standard sequence modalities such as DNA sequences [Dynamic Chunking, Section 4.2, Page 18]."
            )
        elif "moshi" in p_lower or "mimi" in p_lower:
            return (
                "The Moshi architecture employs the Mimi neural audio codec operating at a 12.5 Hz frame rate "
                "with an audio bandwidth of 24 kHz and an overall bitrate of 1.1 kbps [Moshi, Section 3.3, Page 6]. "
                "Mimi utilizes a split Residual Vector Quantization (RVQ) scheme with 8 codebooks for semantic-acoustic "
                "token modeling and maintains an algorithmic latency of 80ms [Moshi, Section 4.1, Page 14]."
            )
        else:
            return (
                "According to the research library, the Transformer and hierarchical sequence models rely on exact "
                "dimension scaling and positional encodings [Attention Is All You Need, Section 3.5, Page 6]. "
                "The specific mechanisms ensure constant path length between sequence elements."
            )
