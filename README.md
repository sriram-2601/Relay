# The Relay — Reproducibility & Codebase Guide

This repository contains the full production-ready codebase, empirical benchmark suite, raw telemetry logs, and write-up for **"The Relay"** ML Engineering Intern assignment at Vaani.

---

## 1. Repository Layout

```text
relay/
├── data/
│   ├── papers/                     # Exact required arXiv research papers
│   │   ├── attention_1706.03762v7.pdf
│   │   ├── moshi_2410.00037v1.pdf
│   │   └── dynamic_chunking_2507.07955v2.pdf
│   └── questions.json             # 18 curated benchmark questions + citations
├── src/
│   ├── ingestion.py               # PDF parser & structural section/page chunker
│   ├── retriever.py               # BM25 + TF-IDF hybrid index with reranking
│   ├── standin_service.py         # Calibrated stand-in model service (Groq LPU simulator)
│   ├── groq_client.py             # Unified API adapter (Real Groq + Stand-in)
│   ├── pipeline.py                # 5-step pipeline with loop & refusal branch
│   ├── evaluator.py               # Table 3 retrieval benchmark runner
│   ├── load_generator.py          # Table 2a/2b load scenario runner
│   └── compute_tables.py          # Derives Tables 1, 2a, 2b, 3 from results.csv
├── scripts/
│   ├── download_papers.py         # Paper downloader from arXiv
│   └── validate_submission.py     # Automated compliance & consistency validator
├── tests/
│   └── test_pipeline.py           # Pipeline integration and refusal test
├── results.csv                    # 156 raw request records with exact metrics
├── writeup.md                     # Complete 2-page write-up (all 8+3 headings)
├── conversation_history.md        # Real assistant interaction trajectory
└── README.md                      # This guide
```

---

## 2. Quick Start & Reproduction

### Prerequisites
- Python 3.10+
- Dependencies: `pip install pymupdf scikit-learn numpy pydantic httpx`

### Step 1: Download Target arXiv Papers
To fetch the exact versions specified in the brief:
```bash
python scripts/download_papers.py
```

### Step 2: Test the Pipeline
Run the end-to-end integration test (tests grounding, citations, and honest refusal):
```bash
python tests/test_pipeline.py
```

### Step 3: Run Retrieval Experiments (Table 3)
Benchmarks the 18 curated questions across 3 chunking/reranking configurations:
```bash
python src/evaluator.py
```

### Step 4: Run Load Scenarios (Table 2a & 2b)
Executes the 5 load shapes (`steady`, `burst`, `sustained overload`, `per-minute limit halved`, `daily budget exhausted`):
```bash
python src/load_generator.py
```

### Step 5: Derive Results Tables
Recomputes all metrics directly from `results.csv`:
```bash
python src/compute_tables.py
```

### Step 6: Validate Everything
Runs the submission consistency checker:
```bash
python scripts/validate_submission.py
```

---

## 3. Using Real Groq Free Tier

To execute queries against real Groq LPUs instead of the calibrated stand-in:
```bash
export GROQ_API_KEY="your-free-groq-api-key"   # Linux/macOS
$env:GROQ_API_KEY="your-free-groq-api-key"     # Windows PowerShell
```
The unified client automatically detects `GROQ_API_KEY` and streams completions from Groq's `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`.

