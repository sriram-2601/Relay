"""
Interactive Demonstration CLI for 'The Relay'.
Allows reviewers to test the 5-step pipeline, inspect live citations,
and trigger a live concurrency burst test from the terminal.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import RelayPipeline
from src.standin_service import CalibratedStandInModel
from src.groq_client import UnifiedModelClient

SAMPLE_QUERIES = [
    ("Attention Is All You Need", "What is the dimension d_model and number of heads h in the Transformer?"),
    ("Moshi (Audio Codec)", "What is the frame rate, bitrate, and algorithmic latency of the Mimi codec?"),
    ("Dynamic Chunking (H-Net)", "How does H-Net eliminate heuristic BPE tokenizers, and on what modality does it achieve a 4x gain?"),
    ("Out-of-Corpus (Strict Refusal)", "What is the quantum fidelity of the entanglement routing algorithm in the library?"),
    ("Subtle Cross-Paper Trap", "What are the recommended hyperparameters for training Whisper large-v3?")
]

def print_banner():
    print("\n" + "=" * 70)
    print("      THE RELAY: 5-STEP GROUNDED RESEARCH ASSISTANT PIPELINE")
    print("             Applicant Submission — Vaani ML Intern")
    print("=" * 70)

def run_single_query(pipeline, question, category):
    print(f"\n[QUERY CATEGORY]: {category}")
    print(f"[INPUT QUESTION]: \"{question}\"")
    print("-" * 70)
    
    t0 = time.time()
    res = pipeline.execute_sync(
        request_id="demo_req",
        question=question,
        scenario="demo",
        force_standin=True
    )
    duration = time.time() - t0
    
    print("\n>>> EXECUTION TRACE (Step-by-Step Latency Decomposition):")
    for s in res.steps_log:
        step_num = s.get("step")
        name = s.get("name")
        dur = s.get("duration_ms", 0)
        extra = ""
        if "branch_decision" in s:
            extra = f" -> Decision: [{s['branch_decision'].upper()}]"
        elif "detected_intent" in s:
            extra = f" -> Target: [{s['detected_intent'].get('target_paper')}]"
        print(f"  Step {step_num}: {name:<36} | {dur:>7.2f} ms {extra}")

    print("\n>>> PIPELINE TELEMETRY:")
    print(f"  Status:             [{res.status.upper()}]")
    print(f"  Grade:              [{res.answer_grade.upper()}]")
    print(f"  Time to First Token: {res.first_token_ms} ms")
    print(f"  Total Latency:       {res.total_ms} ms")
    print(f"  Token Footprint:     {res.tokens_in} in / {res.tokens_out} out")

    print("\n>>> FINAL GROUNDED RESPONSE:")
    print(f"  {res.final_response}")
    print("=" * 70)

def main():
    print_banner()
    print("Initializing RAG indices and pipeline...")
    pipeline = RelayPipeline(retriever_setting="setting_1")
    print("Ready!\n")

    while True:
        print("\nSelect an action:")
        for idx, (cat, q) in enumerate(SAMPLE_QUERIES, 1):
            print(f"  [{idx}] Test Query: {cat}")
        print("  [6] Enter Custom Question")
        print("  [7] Run Live 5-Request Concurrency Spike (Burst Test)")
        print("  [0] Exit")
        
        choice = input("\nEnter choice (0-7): ").strip()
        if choice == "0":
            print("\nExiting. Good luck with the submission!")
            break
        elif choice in [str(i) for i in range(1, 6)]:
            cat, q = SAMPLE_QUERIES[int(choice) - 1]
            run_single_query(pipeline, q, cat)
        elif choice == "6":
            custom_q = input("\nEnter your custom question: ").strip()
            if custom_q:
                run_single_query(pipeline, custom_q, "Custom User Query")
        elif choice == "7":
            print("\nRunning live 5-worker concurrent burst spike...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def worker(i):
                _, q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                return pipeline.execute_sync(f"burst_{i}", q, "burst", force_standin=True)
            
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(5)]
                for idx, f in enumerate(as_completed(futures), 1):
                    r = f.result()
                    print(f"  Completed #{idx}: {r.request_id} | TTFT: {r.first_token_ms}ms | Total: {r.total_ms}ms | Status: {r.status}")
            print(f"\nBurst completed in {time.time()-t_start:.2f}s!")
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

