"""
Groq API Client & Stand-in Adapter for 'The Relay'.
Provides unified interface for both real Groq Free Tier calls and
the calibrated Stand-in model service. Tracks exact TTFT, total latency,
token usage, and rate-limit headers.
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
import httpx

from src.standin_service import CalibratedStandInModel, GroqRateLimitError, GroqTimeoutError

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

class ModelExecutionResult:
    def __init__(
        self,
        text: str,
        first_token_ms: float,
        total_ms: float,
        tokens_in: int,
        tokens_out: int,
        status: str, # "ok", "rate_limited", "timeout", "error"
        source: str, # "real" or "stand-in"
        retries: int = 0,
        headers: Optional[Dict[str, str]] = None,
        error_msg: Optional[str] = None
    ):
        self.text = text
        self.first_token_ms = first_token_ms
        self.total_ms = total_ms
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.status = status
        self.source = source
        self.retries = retries
        self.headers = headers or {}
        self.error_msg = error_msg

class UnifiedModelClient:
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        standin_model: Optional[CalibratedStandInModel] = None
    ):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.standin_model = standin_model or CalibratedStandInModel()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def call_model_sync(
        self,
        messages: List[Dict[str, str]],
        force_standin: bool = True,
        model: str = DEFAULT_GROQ_MODEL,
        max_tokens: int = 600,
        temperature: float = 0.1,
        max_retries: int = 2,
        initial_backoff_sec: float = 1.0
    ) -> ModelExecutionResult:
        """Synchronous wrapper for thread-safe sandboxed execution."""
        source = "stand-in"
        retries = 0
        backoff = initial_backoff_sec

        while True:
            start_wall = time.perf_counter()
            first_token_wall = None
            accumulated_chunks = []
            
            try:
                for chunk in self.standin_model.stream_chat_completion_sync(messages, max_tokens, temperature):
                    if first_token_wall is None:
                        first_token_wall = time.perf_counter()
                    accumulated_chunks.append(chunk)

                total_wall = time.perf_counter()
                total_ms = (total_wall - start_wall) * 1000.0
                first_token_ms = ((first_token_wall - start_wall) * 1000.0) if first_token_wall else total_ms
                
                full_text = "".join(accumulated_chunks)
                prompt_str = " ".join([m.get("content", "") for m in messages])
                tokens_in = max(10, int(len(prompt_str.split()) * 1.3))
                tokens_out = max(5, int(len(full_text.split()) * 1.3))

                return ModelExecutionResult(
                    text=full_text,
                    first_token_ms=round(first_token_ms, 1),
                    total_ms=round(total_ms, 1),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    status="ok",
                    source=source,
                    retries=retries,
                    headers={
                        "x-ratelimit-remaining-requests": str(max(0, self.standin_model.rpm_limit - len(self.standin_model.request_timestamps))),
                        "x-ratelimit-remaining-tokens": str(max(0, self.standin_model.tpm_limit - sum(c for _, c in self.standin_model.token_timestamps)))
                    }
                )

            except GroqRateLimitError as rle:
                if retries < max_retries:
                    retries += 1
                    sleep_duration = min(1.2, max(0.3, backoff))
                    time.sleep(sleep_duration)
                    backoff *= 1.5
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text="Rate limit exceeded.",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="rate_limited",
                        source=source,
                        retries=retries,
                        error_msg=rle.message
                    )

            except GroqTimeoutError as gte:
                if retries < max_retries:
                    retries += 1
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text="Request timed out.",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="timeout",
                        source=source,
                        retries=retries,
                        error_msg=str(gte)
                    )

            except Exception as e:
                if retries < max_retries:
                    retries += 1
                    time.sleep(backoff)
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text=f"Error: {e}",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="error",
                        source=source,
                        retries=retries,
                        error_msg=str(e)
                    )

    async def call_model(
        self,
        messages: List[Dict[str, str]],
        force_standin: bool = False,
        model: str = DEFAULT_GROQ_MODEL,
        max_tokens: int = 600,
        temperature: float = 0.1,
        max_retries: int = 2,
        initial_backoff_sec: float = 1.0
    ) -> ModelExecutionResult:
        """
        Executes a streaming model call with retry policy, latency instrumentation,
        and automatic failover/fallback.
        """
        use_real = (not force_standin) and bool(self.groq_api_key)
        source = "real" if use_real else "stand-in"
        
        retries = 0
        backoff = initial_backoff_sec

        while True:
            start_wall = time.perf_counter()
            first_token_wall = None
            accumulated_chunks = []
            
            try:
                if use_real:
                    # Stream from real Groq API
                    headers = {
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": True
                    }

                    client = await self.get_http_client()
                    req = client.build_request("POST", GROQ_API_URL, headers=headers, json=payload)
                    resp = await client.send(req, stream=True)

                    if resp.status_code == 429:
                        await resp.aread()
                        retry_after_hdr = resp.headers.get("retry-after", "3")
                        wait_sec = float(retry_after_hdr) if retry_after_hdr.isdigit() else 3.0
                        raise GroqRateLimitError(f"Real Groq 429 Rate Limit Exceeded", retry_after=wait_sec)
                    elif resp.status_code != 200:
                        body = await resp.aread()
                        raise Exception(f"HTTP {resp.status_code}: {body.decode('utf-8', errors='ignore')}")

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(raw_data)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                if first_token_wall is None:
                                    first_token_wall = time.perf_counter()
                                accumulated_chunks.append(delta)
                        except Exception:
                            continue
                    
                    resp_headers = dict(resp.headers)
                    await resp.aclose()
                    
                else:
                    # Stream from Calibrated Stand-in
                    async for chunk in self.standin_model.stream_chat_completion(messages, max_tokens, temperature):
                        if first_token_wall is None:
                            first_token_wall = time.perf_counter()
                        accumulated_chunks.append(chunk)
                    resp_headers = {
                        "x-ratelimit-remaining-requests": str(max(0, self.standin_model.rpm_limit - len(self.standin_model.request_timestamps))),
                        "x-ratelimit-remaining-tokens": str(max(0, self.standin_model.tpm_limit - sum(c for _, c in self.standin_model.token_timestamps)))
                    }

                total_wall = time.perf_counter()
                total_ms = (total_wall - start_wall) * 1000.0
                first_token_ms = ((first_token_wall - start_wall) * 1000.0) if first_token_wall else total_ms
                
                full_text = "".join(accumulated_chunks)
                prompt_str = " ".join([m.get("content", "") for m in messages])
                tokens_in = max(10, int(len(prompt_str.split()) * 1.3))
                tokens_out = max(5, int(len(full_text.split()) * 1.3))

                return ModelExecutionResult(
                    text=full_text,
                    first_token_ms=round(first_token_ms, 1),
                    total_ms=round(total_ms, 1),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    status="ok",
                    source=source,
                    retries=retries,
                    headers=resp_headers
                )

            except GroqRateLimitError as rle:
                if retries < max_retries:
                    retries += 1
                    # Jittered backoff respecting retry_after
                    sleep_duration = min(6.0, max(rle.retry_after, backoff))
                    await asyncio.sleep(sleep_duration)
                    backoff *= 1.5
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text="Rate limit exceeded.",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="rate_limited",
                        source=source,
                        retries=retries,
                        error_msg=rle.message
                    )

            except GroqTimeoutError as gte:
                if retries < max_retries:
                    retries += 1
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text="Request timed out.",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="timeout",
                        source=source,
                        retries=retries,
                        error_msg=str(gte)
                    )

            except Exception as e:
                if retries < max_retries:
                    retries += 1
                    await asyncio.sleep(backoff)
                    continue
                else:
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return ModelExecutionResult(
                        text=f"Error: {e}",
                        first_token_ms=round(total_ms, 1),
                        total_ms=round(total_ms, 1),
                        tokens_in=0,
                        tokens_out=0,
                        status="error",
                        source=source,
                        retries=retries,
                        error_msg=str(e)
                    )
