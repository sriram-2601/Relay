### Table 1: Limits you found in the real service

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute | 30 RPM | Groq docs + 429 response body header inspection | High |
| Tokens per minute | 12,000 TPM (Llama 3.3 70B) | Groq developer console rate-limit table | High |
| Requests per day | 1,000 RPD (Llama 3.3 70B) | Account limits documentation | High |
| Tokens per day | 100,000 TPD (Llama 3.3 70B) | Groq platform quotas dashboard | High |
| Requests in flight at once | 5 concurrent requests | Empirical probe; latency jumps quadratically past 5 | Medium |
| What it returns when a limit is hit | HTTP 429 with `code: rate_limit_exceeded` & `retry-after` header | Live probe payload analysis | High |
| Time until it recovers | 6s – 18s (sliding minute); up to 24h at 00:00 UTC for daily cap | Measured recovery from `retry-after` header | High |

### Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady | stand-in | 15 | 23.3 | 38.6 | 362 / 397 / 413 | 1545 / 1751 / 2042 |
| burst | stand-in | 25 | 2.7 | 550.7 | 988 / 1502 / 1526 | 2211 / 2678 / 2713 |
| sustained overload | stand-in | 30 | 2.3 | 769.7 | 756 / 991 / 1125 | 2076 / 2218 / 2304 |
| per-minute limit halved | stand-in | 20 | 2.2 | 540.3 | 544 / 747 / 777 | 2209 / 2214 / 2219 |
| daily budget exhausted | stand-in | 12 | 2.2 | 322.3 | 0 / 0 / 0 | 2212 / 2223 / 2231 |


### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady | 100.0% | 0.0% | 0.0% | 0.07 | 1 | 2036 | $1.217 |
| burst | 60.0% | 40.0% | 0.0% | 0.8 | 15 | 1861 | $1.114 |
| sustained overload | 60.0% | 40.0% | 0.0% | 0.8 | 8 | 1665 | $0.998 |
| per-minute limit halved | 45.0% | 55.0% | 0.0% | 1.1 | 6 | 1478 | $0.887 |
| daily budget exhausted | 0.0% | 100.0% | 0.0% | 2.0 | 1 | 0 | $0.000 |


### Table 3: Retrieval settings (at least three)

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Small chunks (200t), k=3, no rerank | 18 | 77.8% | 11.1% | 11.1% | 721 | 247 / 272 | 1420 / 1455 |
| Medium chunks (450t), k=5, no rerank | 18 | 72.2% | 16.7% | 11.1% | 2058 | 379 / 409 | 1555 / 1585 |
| Medium chunks (450t), k=8, rerank to 4 | 18 | 72.2% | 16.7% | 11.1% | 1651 | 325 / 364 | 1506 / 1546 |

