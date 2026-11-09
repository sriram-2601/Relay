"""
Evaluation Module for 'The Relay'.
Benchmarks the 18 curated questions across the 3 distinct retrieval settings.
Logs every request row directly to results.csv.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import csv
import time
from typing import List, Dict, Any

from src.pipeline import RelayPipeline, PipelineOutput
from src.standin_service import CalibratedStandInModel
from src.groq_client import UnifiedModelClient

RESULTS_CSV_PATH = "results.csv"
CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

def init_results_csv():
    """Initializes results.csv with header if not present."""
    if not os.path.exists(RESULTS_CSV_PATH):
        with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def append_to_csv(output: PipelineOutput):
    """Appends a single pipeline execution record to results.csv."""
    with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            output.request_id,
            output.scenario,
            output.source,
            output.start_time,
            output.first_token_ms,
            output.total_ms,
            output.status,
            output.retries,
            output.tokens_in,
            output.tokens_out,
            output.retrieval_setting,
            output.answer_grade
        ])

def run_retrieval_benchmarks() -> Dict[str, List[PipelineOutput]]:
    """Runs all 18 test questions under the three retrieval settings."""
    init_results_csv()
    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    settings = ["setting_1", "setting_2", "setting_3"]
    all_results = {}

    for s_name in settings:
        print(f"\n=======================================================")
        print(f"Running Retrieval Benchmark: {s_name.upper()}")
        print(f"=======================================================")
        # Use clean calibrated stand-in with unconstrained TPM for retrieval quality measurement
        standin = CalibratedStandInModel(rpm_limit=120, tpm_limit=500000, failure_rate=0.0)
        client = UnifiedModelClient(standin_model=standin)
        pipeline = RelayPipeline(retriever_setting=s_name, model_client=client)
        setting_outputs = []

        for idx, q_item in enumerate(questions, 1):
            req_id = f"eval_{s_name}_q{idx:02d}"
            print(f"[{idx}/{len(questions)}] {req_id}: {q_item['question'][:60]}...")
            
            output = pipeline.execute_sync(
                request_id=req_id,
                question=q_item["question"],
                scenario=f"retrieval_{s_name}",
                force_standin=True,
                expected_refusal=q_item["expected_refusal"]
            )
            append_to_csv(output)
            setting_outputs.append(output)
            print(f"   -> Status: {output.status} | Grade: {output.answer_grade} | TTFT: {output.first_token_ms}ms | Total: {output.total_ms}ms")
            
            # Short pacing between evaluation requests
            time.sleep(0.05)

        all_results[s_name] = setting_outputs

    return all_results

if __name__ == "__main__":
    results = run_retrieval_benchmarks()
    print("\nRetrieval benchmarks completed successfully. Rows appended to results.csv.")
