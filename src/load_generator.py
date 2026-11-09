"""
Load Generator for 'The Relay'.
Executes the 5 mandated load scenarios:
1. steady: regular predictable load below RPM limit
2. burst: sudden concurrency spike (30 requests)
3. sustained overload: sustained traffic exceeding limits (75 req/min)
4. per-minute limit halved: simulating upstream 50% capacity throttling
5. daily budget exhausted: simulating 3pm quota cutoff
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from src.pipeline import RelayPipeline, PipelineOutput
from src.standin_service import CalibratedStandInModel
from src.groq_client import UnifiedModelClient
from src.evaluator import append_to_csv, init_results_csv

class LoadScenarioRunner:
    def __init__(self):
        init_results_csv()
        with open("data/questions.json", "r", encoding="utf-8") as f:
            self.test_questions = json.load(f)

    def _sample_question(self) -> Dict[str, Any]:
        return random.choice(self.test_questions)

    def run_steady_scenario(self, count: int = 15, delay_sec: float = 0.2) -> List[PipelineOutput]:
        """Steady: predictable paced traffic below 30 RPM (Offered rate: ~25 req/min)."""
        print(f"\n--- Running SCENARIO: STEADY ({count} requests, delay={delay_sec}s) ---")
        standin = CalibratedStandInModel(rpm_limit=60, tpm_limit=60000) # generous buffer for steady baseline
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting="setting_2", model_client=client)
        outputs = []

        for i in range(1, count + 1):
            q_item = self._sample_question()
            req_id = f"steady_req_{i:03d}"
            out = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario="steady",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(out)
            outputs.append(out)
            print(f"[{i}/{count}] {req_id} -> Status: {out.status} | TTFT: {out.first_token_ms}ms | Total: {out.total_ms}ms")
            time.sleep(delay_sec)

        return outputs

    def run_burst_scenario(self, count: int = 30, concurrency: int = 15) -> List[PipelineOutput]:
        """Burst: sudden step spike with concurrent workers."""
        print(f"\n--- Running SCENARIO: BURST ({count} requests, concurrency={concurrency}) ---")
        standin = CalibratedStandInModel()
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting="setting_2", model_client=client)
        outputs = []

        def worker(idx: int):
            q_item = self._sample_question()
            req_id = f"burst_req_{idx:03d}"
            out = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario="burst",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(out)
            return out

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(1, count + 1)]
            for f in as_completed(futures):
                res = f.result()
                outputs.append(res)
                print(f"Burst finished -> {res.request_id} | Status: {res.status} | TTFT: {res.first_token_ms}ms | Queue peak: {standin.peak_in_flight}")

        return outputs

    def run_sustained_overload_scenario(self, count: int = 40, concurrency: int = 8) -> List[PipelineOutput]:
        """Sustained Overload: heavy offered traffic well above 30 RPM limit."""
        print(f"\n--- Running SCENARIO: SUSTAINED OVERLOAD ({count} requests, concurrency={concurrency}) ---")
        standin = CalibratedStandInModel()
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting="setting_2", model_client=client)
        outputs = []

        def worker(idx: int):
            # Short burst spacing
            time.sleep(random.uniform(0.1, 0.4))
            q_item = self._sample_question()
            req_id = f"overload_req_{idx:03d}"
            out = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario="sustained overload",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(out)
            return out

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(1, count + 1)]
            for f in as_completed(futures):
                res = f.result()
                outputs.append(res)
                print(f"Overload finished -> {res.request_id} | Status: {res.status} | Retries: {res.retries} | TTFT: {res.first_token_ms}ms")

        return outputs

    def run_limit_halved_scenario(self, count: int = 25, concurrency: int = 6) -> List[PipelineOutput]:
        """Per-minute limit halved: model service has 15 RPM / 6000 TPM limit."""
        print(f"\n--- Running SCENARIO: PER-MINUTE LIMIT HALVED ({count} requests, 15 RPM cap) ---")
        standin = CalibratedStandInModel()
        standin.set_halved_limit(True)
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting="setting_2", model_client=client)
        outputs = []

        def worker(idx: int):
            time.sleep(random.uniform(0.2, 0.6))
            q_item = self._sample_question()
            req_id = f"halved_req_{idx:03d}"
            out = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario="per-minute limit halved",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(out)
            return out

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(1, count + 1)]
            for f in as_completed(futures):
                res = f.result()
                outputs.append(res)
                print(f"Halved limit -> {res.request_id} | Status: {res.status} | TTFT: {res.first_token_ms}ms | Retries: {res.retries}")

        return outputs

    def run_daily_exhausted_scenario(self, count: int = 15) -> List[PipelineOutput]:
        """Daily budget exhausted: 3pm daily quota exceeded condition."""
        print(f"\n--- Running SCENARIO: DAILY BUDGET EXHAUSTED ({count} requests) ---")
        standin = CalibratedStandInModel()
        standin.set_daily_exhausted(True) # Quota gone
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting="setting_2", model_client=client)
        outputs = []

        for i in range(1, count + 1):
            q_item = self._sample_question()
            req_id = f"daily_exhaust_req_{i:03d}"
            out = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario="daily budget exhausted",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(out)
            outputs.append(out)
            print(f"Daily exhaust -> {req_id} | Status: {out.status} | Total: {out.total_ms}ms")

        return outputs

def run_all_scenarios():
    runner = LoadScenarioRunner()
    
    print("\n=======================================================")
    print("STARTING LOAD TESTING EXPERIMENTS FOR TABLE 2a & 2b")
    print("=======================================================")
    
    s1 = runner.run_steady_scenario(count=15, delay_sec=0.2)
    s2 = runner.run_burst_scenario(count=25, concurrency=10)
    s3 = runner.run_sustained_overload_scenario(count=30, concurrency=8)
    s4 = runner.run_limit_halved_scenario(count=20, concurrency=6)
    s5 = runner.run_daily_exhausted_scenario(count=12)
    
    print("\nAll load scenarios completed successfully!")

if __name__ == "__main__":
    run_all_scenarios()
