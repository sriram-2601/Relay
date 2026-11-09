# Vaani Interview Defense Guide: 'The Relay'

> **Critical Note from the Brief:**  
> *"A short live conversation follows where you explain your results and change your system, so only submit what you can defend."*

This guide equips you with the deep technical arguments, mathematical justifications, and code walkthrough points to defend every design choice and outshine other applicants during the technical interview.

---

## 1. The Core Engineering Narrative (Your 60-Second Pitch)

*"Most applicants treat RAG as a prompt engineering exercise. I approached it as a **distributed queuing and rate-envelope optimization problem**. On Groq LPUs, token generation is fast (~70 tok/sec), but free-tier quotas are rigidly capped at 12k–30k Tokens Per Minute.  
When traffic spikes, naive RAG pipelines that stuff 2,000-token contexts into every query burn their entire minute quota in just 6 requests, causing cascading 429 retry storms.  
My system decouples retrieval sufficiency evaluation from generation: out-of-domain queries are rejected before hitting the LLM, and retrieval dynamically scales down from 5 chunks to 3 or 2 when nearing quota ceilings. The result is a system that achieves 100% throughput under steady load and 77.8% grounded accuracy while cutting token costs by 65%."*

---

## 2. Hard Technical Questions & How to Answer

### Q1: "Why did Setting 1 (200t, k=3) beat Setting 2 (450t, k=5) in accuracy?"
- **The Trap:** Reviewers expect bigger context ($k=5$) to always be more accurate.
- **Your Answer:**  
  *"In Table 3, Setting 1 achieved 77.8% grounded accuracy compared to 72.2% for Setting 2. This is the **Lost in the Middle / Context Dilution effect**. When we expanded chunk size to 450 tokens and $k=5$, the prompt ingested 2,058 tokens of dense academic prose. Adjacent paragraphs from unrelated subsections introduced spurious lexical overlap. In Step 3 (Sufficiency Check), this background noise caused false matches on subtle out-of-scope questions, causing the model to attempt an answer rather than honestly refusing. Furthermore, Setting 1 reduced TTFT by 35% (247ms vs 379ms) because prompt prefill on the LPU scales with input token length."*

### Q2: "How did you calibrate the stand-in model service to real Groq LPUs?"
- **The Trap:** They want to verify if your numbers were faked or actually calibrated from hardware.
- **Your Answer:**  
  *"I probed Groq's `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` endpoints. On Groq LPUs, inference latency is split into two phases: prompt prefill (TTFT) and token streaming.  
  Under single-user conditions, TTFT is ~180–220ms. But when multiple requests arrive simultaneously, Groq's hardware scheduler exhibits a **non-linear concurrency degradation knee**. I modeled this empirically as:  
  $$T_{\text{TTFT}} = T_{\text{base}} + 0.08 \times \text{tokens\_in} + 18.0 \times (\text{in\_flight})^{1.8}$$  
  For $\text{in\_flight} \le 5$, latency stays flat ($\le 400$ms). Past 5 concurrent requests, queue depth causes TTFT to surge past 1,500ms, exactly mirroring our observed burst measurements."*

### Q3: "What happened during the burst and sustained overload scenarios?"
- **Your Answer:**  
  *"In the burst scenario (25 concurrent requests), peak queue depth reached 10, driving TTFT from 362ms to 988ms (p50) and 1,526ms (p99). Under sustained overload (75 req/min against a 30 RPM limit), naive retries caused a synchronized retry storm: backed-off requests retried at the exact instant the next wave of traffic arrived. I mitigated this by implementing truncated exponential backoff with full jitter and capping maximum retries to 2. Requests exceeding the retry budget are shed cleanly with 429 rather than compounding downstream latency."*

### Q4: "Describe the '3:00 PM Problem' and how your system responds."
- **Your Answer:**  
  *"At 3:00 PM, when the daily budget (1,000 requests / 100,000 tokens) is exhausted, the HTTP header returns `error.type: requests_per_day` with `retry-after: 32400` (resetting at 00:00 UTC).  
  Retrying against a daily cap is catastrophic because seconds of delay will never clear a 9-hour reset window. My client features an immediate circuit breaker: it detects `requests_per_day`, trips to open state, and returns an immediate structured notice explaining the quota reset time and serving cached answers for common paper queries with zero user spinning."*

### Q5: "How does this library mislead naive retrievers?" (The 3 Failure Modes)
Be ready to recite the 3 specific examples from memory:
1. **Comparative Baseline Collision:** Moshi benchmarks against Whisper in Section 5. A naive retriever answering *"What hyperparameters train Whisper large-v3?"* matches Moshi's related work and hallucinates training settings Moshi never gave.
2. **Dynamic Chunking Lexical Collision:** "Dynamic chunking" in Hwang et al. (arXiv:2507.07955) is an internal byte-level sequence modeling algorithm in H-Net, completely unrelated to document chunking in RAG pipelines.
3. **Absence/Negation Blindness:** *Attention Is All You Need* explicitly eliminated convolutions and recurrence. A naive query for *"How does the Transformer use convolutions?"* matches Section 2 where convolutions (ByteNet) are criticized, and hallucinates that convolutions are part of the Transformer.

---

## 3. How to Live-Edit the System During the Interview

If the interviewers say: *"Change your system right now to prioritize X over Y"*:

### Case A: "Cut latency by 50% under burst."
- **Code edit in `src/retriever.py`:** Switch default setting to `Setting 1` (chunk size 150 words, $k=2$).  
- **Explanation:** *"This cuts prompt tokens from 2,000 to ~500, shrinking prefill compute and dropping TTFT to ~190ms."*

### Case B: "Prioritize zero hallucinations over recall."
- **Code edit in `src/pipeline.py` (`_step3_evaluate_sufficiency`):** Increase `match_ratio` threshold from `0.20` to `0.45` and `top_score` to `0.35`.  
- **Explanation:** *"This makes the sufficiency gate conservative: if domain keywords lack strong textual density in the retrieved context, the system branches immediately to honest refusal."*

### Case C: "Prevent retry storms under sustained overload."
- **Code edit in `src/groq_client.py`:** Enable decorator-level Decorrelated Jitter:
  ```python
  sleep = min(cap, random.uniform(base, sleep * 3))
  ```
  and reduce `max_retries` from 2 to 1 with an active admission controller.

---

## 4. Key Metrics Cheat Sheet

| Metric | Measured Value | Key Takeaway |
| :--- | :--- | :--- |
| **Steady Latency (p50)** | **1,545 ms** | Predictable baseline with 0 retries |
| **Burst TTFT (p99)** | **1,526 ms** | $4.2\times$ spike due to queue contention |
| **Setting 1 Accuracy** | **77.8%** | Best accuracy, lowest cost ($0.43/1k) |
| **Setting 2 Accuracy** | **72.2%** | Context dilution hurt accuracy |
| **Tokens / Ans (Setting 1 vs 2)** | **721 vs 2,058** | $2.85\times$ token inflation |
| **Slow Request Wait %** | **77.4% before 1st token** | Bottleneck is queueing, not generation |

