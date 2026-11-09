import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from src.pipeline import RelayPipeline

def main():
    print("1. Initializing pipeline...")
    t0 = time.time()
    p = RelayPipeline("setting_2")
    print(f"Pipeline initialized in {time.time()-t0:.2f}s.")

    print("\n2. Executing pipeline query (test_01)...")
    t0 = time.time()
    res = p.execute_sync(
        request_id="test_01",
        question="What is the dimension d_model and number of heads h in Attention Is All You Need?",
        scenario="steady",
        force_standin=True
    )
    print(f"Pipeline executed in {time.time()-t0:.2f}s!")
    print(f"Status: {res.status}")
    print(f"TTFT (ms): {res.first_token_ms}")
    print(f"Total time (ms): {res.total_ms}")
    print(f"Tokens in/out: {res.tokens_in} / {res.tokens_out}")
    print(f"Grade: {res.answer_grade}")
    print(f"Response:\n{res.final_response}")
    print("\nSteps log:")
    for s in res.steps_log:
        print(f" - Step {s.get('step')}: {s.get('name')} ({s.get('duration_ms')}ms)")

    print("\n3. Testing Refusal branch with an unanswerable question...")
    res_refusal = p.execute_sync(
        request_id="test_refusal_01",
        question="What are the details of the quantum teleportation protocol in the papers?",
        scenario="steady",
        force_standin=True,
        expected_refusal=True
    )
    print(f"Refusal status: {res_refusal.status}")
    print(f"Refusal grade: {res_refusal.answer_grade}")
    print(f"Refusal response:\n{res_refusal.final_response}")

if __name__ == "__main__":
    main()
