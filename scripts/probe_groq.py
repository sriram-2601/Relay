"""
Live Groq API Probing Script for 'The Relay'.
Verifies your Groq API key, checks rate limits, inspects HTTP response headers,
and records empirical baseline latency (TTFT and tokens/sec).
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

def probe_groq(api_key: str, model: str = "llama-3.3-70b-versatile"):
    print(f"\n" + "=" * 65)
    print(f"PROBING LIVE GROQ API: Model = {model}")
    print("=" * 65)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "In one sentence, what is the key advantage of self-attention in the Transformer "
        "according to 'Attention Is All You Need'?"
    )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert ML research assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "stream": True
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    accumulated_tokens = []
    
    with httpx.Client(timeout=30.0) as client:
        try:
            with client.stream("POST", GROQ_API_URL, headers=headers, json=payload) as resp:
                print(f"HTTP Status: {resp.status_code}")
                
                # Capture Rate Limit Headers
                rl_headers = {
                    k: v for k, v in resp.headers.items() 
                    if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
                }
                
                print("\n--- Live Groq Rate Limit Headers ---")
                for k, v in rl_headers.items():
                    print(f"  {k}: {v}")

                if resp.status_code == 429:
                    print("\n[WARNING]: 429 Rate Limit Encountered on Live Probe!")
                    return
                elif resp.status_code != 200:
                    print(f"\n[ERROR]: Request failed with code {resp.status_code}")
                    return

                # Stream response
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(raw_data)
                        delta = chunk_json["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            accumulated_tokens.append(delta)
                    except Exception:
                        continue

            total_time = time.perf_counter() - start_time
            ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else total_time * 1000.0
            total_ms = total_time * 1000.0
            
            output_text = "".join(accumulated_tokens)
            tokens_out = int(len(output_text.split()) * 1.3)
            speed = tokens_out / max(0.01, (total_time - (first_token_time - start_time))) if first_token_time else 0.0

            print("\n--- Live Performance Metrics ---")
            print(f"  Time to First Token (TTFT): {ttft_ms:.1f} ms")
            print(f"  Total Latency:              {total_ms:.1f} ms")
            print(f"  Tokens Generated:           ~{tokens_out} tokens")
            print(f"  Generation Speed:           ~{speed:.1f} tokens/sec")
            print(f"\n--- Model Response ---")
            print(f"  \"{output_text.strip()}\"")
            print("=" * 65)

            # Save empirical probe log
            os.makedirs("data", exist_ok=True)
            probe_record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "ttft_ms": round(ttft_ms, 1),
                "total_ms": round(total_ms, 1),
                "tokens_out": tokens_out,
                "speed_tok_sec": round(speed, 1),
                "rate_limit_headers": rl_headers,
                "response_sample": output_text.strip()
            }
            with open("data/groq_live_probe.json", "w", encoding="utf-8") as f:
                json.dump(probe_record, f, indent=2)
            print("Successfully saved live probe telemetry to data/groq_live_probe.json!")

        except Exception as e:
            print(f"\n[ERROR] Probing failed: {e}")

def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("\n[!] GROQ_API_KEY environment variable is not set.")
        print("Please enter your free Groq API key (starts with 'gsk_...'):")
        key = input("Enter API Key: ").strip()
        if not key:
            print("No key provided. Exiting.")
            return

    probe_groq(key, model="llama-3.3-70b-versatile")
    probe_groq(key, model="llama-3.1-8b-instant")

if __name__ == "__main__":
    main()

