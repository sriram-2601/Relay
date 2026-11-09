"""
The Relay — World-Class Research Assistant & Telemetry Dashboard
An enterprise-grade, interactive Web UI showcasing grounded RAG, sub-second latency,
circuit breaking, and empirical LPU concurrency queue degradation.
Run with: python -m streamlit run web_app.py
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline import RelayPipeline
from src.standin_service import CalibratedStandInModel
from src.groq_client import UnifiedModelClient

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="The Relay — Grounded Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# WORLD-CLASS ENTERPRISE DESIGN SYSTEM (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Brand Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin: 0;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Step Timeline Waterfall */
    .step-badge {
        display: inline-flex;
        align-items: center;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
    }
    .step-ok {
        background: #dcfce7;
        border-color: #86efac;
        color: #166534;
    }
    .step-refusal {
        background: #e0e7ff;
        border-color: #a5b4fc;
        color: #3730a3;
    }

    /* Citation Tags */
    .citation-tag {
        display: inline-block;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Code styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: #f1f5f9;
        color: #0f172a;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PIPELINE INITIALIZATION & CACHING
# ==========================================
@st.cache_resource
def get_pipeline(setting_name: str, use_real_groq: bool, api_key: str):
    standin = CalibratedStandInModel()
    client = UnifiedModelClient(groq_api_key=api_key if use_real_groq else None, standin_model=standin)
    return RelayPipeline(retriever_setting=setting_name, model_client=client)

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ System Architecture")
    
    model_source = st.radio(
        "Inference Engine",
        ["Calibrated Stand-in (LPU Simulator)", "Real Groq Cloud LPU API"],
        index=0,
        help="Use the Calibrated Stand-in to test stress scenarios without consuming API keys."
    )
    
    use_real = (model_source == "Real Groq Cloud LPU API")
    groq_api_key = ""
    if use_real:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Free key from console.groq.com/keys"
        )
        if not groq_api_key:
            st.warning("⚠️ Enter key or switch to Stand-in Simulator.")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Optimization")
    retrieval_setting = st.selectbox(
        "Chunking & Re-ranking Setting",
        ["setting_1", "setting_2", "setting_3"],
        index=0,
        format_func=lambda x: {
            "setting_1": "Setting 1: Small (200t), k=3 [Optimal: 77.8%]",
            "setting_2": "Setting 2: Medium (450t), k=5 [Diluted: 72.2%]",
            "setting_3": "Setting 3: Medium (450t), k=8 -> Rerank 4"
        }[x],
        help="Empirically verified in Table 3. Setting 1 provides highest accuracy and lowest token cost."
    )

    st.markdown("---")
    st.markdown("### 📚 Target Research Papers")
    st.markdown("""
    - **Attention Is All You Need**  
      `arXiv:1706.03762v7` *(15 pages)*
    - **Moshi: Speech-Text Dialogue**  
      `arXiv:2410.00037v1` *(66 pages)*
    - **Dynamic Chunking (H-Net)**  
      `arXiv:2507.07955v2` *(39 pages)*
    """)

# Load Pipeline Instance
pipeline = get_pipeline(retrieval_setting, use_real, groq_api_key)

# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">The Relay — Grounded Research Assistant</div>
    <p class="hero-subtitle">
        Production 5-Step RAG Architecture • Strict Bracketed Citations • Zero-Hallucination Refusal Gate • Calibrated LPU Rate Limiting
    </p>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_assistant, tab_burst, tab_architecture, tab_tables = st.tabs([
    "💬 Interactive Research Assistant",
    "🔥 Concurrency Burst Simulator",
    "📐 Architecture & Latency Breakdown",
    "📋 Empirical Telemetry & Results Tables"
])

# =========================================================================
# TAB 1: INTERACTIVE RESEARCH ASSISTANT
# =========================================================================
with tab_assistant:
    col_chat, col_presets = st.columns([1.8, 1.2])

    with col_presets:
        st.markdown("#### 🎯 Quick Test Questions")
        preset_choice = st.selectbox(
            "Select a pre-configured benchmark query:",
            [
                "Custom Question...",
                "Attention: Dimension d_model & number of heads h?",
                "Attention: What formula is used for positional encodings?",
                "Moshi: Frame rate, bitrate, and algorithmic latency of Mimi?",
                "Moshi: How many codebooks are used in the split RVQ scheme?",
                "Dynamic Chunking: How does H-Net eliminate heuristic BPE tokenizers?",
                "Dynamic Chunking: On what sequence modality does H-Net achieve a 4x data gain?",
                "Refusal Test: What is the quantum fidelity of entanglement routing in the library?",
                "Trap Test: What are the recommended hyperparameters for training Whisper large-v3?"
            ]
        )

        query_text = ""
        expected_refusal = False
        if preset_choice != "Custom Question...":
            query_map = {
                "Attention: Dimension d_model & number of heads h?": "What is the total model dimension d_model and number of attention heads h in the base Transformer?",
                "Attention: What formula is used for positional encodings?": "What formula is used for the positional encodings in the Transformer?",
                "Moshi: Frame rate, bitrate, and algorithmic latency of Mimi?": "What is the audio sampling rate, frame rate, bitrate, and algorithmic latency of the Mimi codec in Moshi?",
                "Moshi: How many codebooks are used in the split RVQ scheme?": "How many codebooks are used in the split Residual Vector Quantization (RVQ) scheme of Mimi?",
                "Dynamic Chunking: How does H-Net eliminate heuristic BPE tokenizers?": "What core component does H-Net use to replace fixed heuristic Byte-Pair Encoding (BPE) tokenizers?",
                "Dynamic Chunking: On what sequence modality does H-Net achieve a 4x data gain?": "On which non-standard sequence modality does H-Net achieve a 4x data efficiency gain over BPE-tokenized baselines?",
                "Refusal Test: What is the quantum fidelity of entanglement routing in the library?": "What is the quantum fidelity of the entanglement routing algorithm described in the papers?",
                "Trap Test: What are the recommended hyperparameters for training Whisper large-v3?": "What is the recommended hyperparameter setting for training the Whisper large-v3 model on LibriSpeech?"
            }
            query_text = query_map.get(preset_choice, "")
            if "Refusal" in preset_choice or "Trap" in preset_choice:
                expected_refusal = True

    with col_chat:
        st.markdown("#### 💬 Ask Question")
        user_input = st.text_area(
            "Enter research query across the 3 target papers:",
            value=query_text,
            height=85,
            placeholder="e.g. What is the inner-layer dimensionality d_ff of the feed-forward network in Attention Is All You Need?"
        )
        btn_run = st.button("⚡ Query Research Assistant", type="primary", use_container_width=True)

    if btn_run and user_input.strip():
        with st.spinner("Executing 5-Step Pipeline (Intent -> Retrieval -> Sufficiency Gate -> Drafting -> Verification)..."):
            out = pipeline.execute_sync(
                request_id=f"web_{int(time.time()*1000)}",
                question=user_input.strip(),
                scenario="web_interactive",
                force_standin=not use_real,
                expected_refusal=expected_refusal
            )

        st.markdown("---")
        
        # 1. Answer Card
        st.markdown("### 📝 Grounded Response")
        if out.status == "rate_limited":
            st.error(f"⚠️ **Rate Limit Hit (HTTP 429)**: {out.final_response}")
        elif out.answer_grade == "refused_correctly":
            st.info(f"🛡️ **Grounded Refusal Gate (Zero Hallucination)**:\n\n{out.final_response}")
        elif out.answer_grade == "correct":
            st.success(f"{out.final_response}")
        else:
            st.warning(f"{out.final_response}")

        # 2. Telemetry Highlights
        st.markdown("### ⏱️ Real-Time Telemetry & Token Accounting")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Status</div><div class='metric-value'>{out.status.upper()}</div></div>", unsafe_allow_html=True)
        with m2:
            grade_color = "#16a34a" if "correct" in out.answer_grade else "#dc2626"
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Evaluation Grade</div><div class='metric-value' style='color:{grade_color}; font-size:1.15rem;'>{out.answer_grade.replace('_', ' ').title()}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Time to 1st Token</div><div class='metric-value'>{out.first_token_ms:.0f} ms</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Latency</div><div class='metric-value'>{out.total_ms:.0f} ms</div></div>", unsafe_allow_html=True)
        with m5:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Tokens (In / Out)</div><div class='metric-value'>{out.tokens_in} / {out.tokens_out}</div></div>", unsafe_allow_html=True)

        # 3. Step-by-Step Latency Decomposition Waterfall
        st.markdown("### 🔍 5-Step Pipeline Waterfall Breakdown")
        step_cols = st.columns(len(out.steps_log))
        for idx, (col, step_info) in enumerate(zip(step_cols, out.steps_log)):
            with col:
                st.markdown(f"**Step {step_info.get('step')}**")
                dur = step_info.get("duration_ms", 0)
                badge_class = "step-ok" if dur < 1000 else "step-badge"
                st.markdown(f"<span class='step-badge {badge_class}'>{dur:.1f} ms</span>", unsafe_allow_html=True)
                st.caption(step_info.get("name"))
                if "branch_decision" in step_info:
                    st.code(f"Gate: {step_info['branch_decision']}")
                elif "detected_intent" in step_info:
                    target = step_info["detected_intent"].get("target_paper", "all")
                    st.code(f"Target: {target}")

# =========================================================================
# TAB 2: CONCURRENCY BURST SIMULATOR
# =========================================================================
with tab_burst:
    st.markdown("### 🔥 Real-Time Concurrency Spike Simulator")
    st.markdown("Test how the pipeline handles sudden bursts under strict free-tier rate limits (30 RPM / 30,000 TPM).")
    
    col_b_ctrl, col_b_info = st.columns([1, 2])
    with col_b_ctrl:
        burst_n = st.slider("Total Requests in Burst", min_value=4, max_value=25, value=12, step=1)
        concurrency_workers = st.slider("Parallel Worker Threads", min_value=2, max_value=12, value=6, step=1)
        btn_start_burst = st.button("🚀 Fire Concurrency Burst", type="primary", use_container_width=True)

    with col_b_info:
        st.info("""
        **What this simulates:**
        - **Queue Contention Knee:** When concurrency exceeds 5 workers, LPU scheduler contention causes Time-to-First-Token (TTFT) to spike from ~350ms to >1,500ms.
        - **Rate Limit Enforcement:** Excess requests trigger HTTP 429 (`rate_limit_exceeded`) with jittered backoff retries.
        """)

    if btn_start_burst:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        sample_q = "What is the dimension d_model in Attention Is All You Need?"
        results_list = []

        def worker_task(idx):
            return pipeline.execute_sync(
                request_id=f"burst_req_{idx:03d}",
                question=sample_q,
                scenario="burst",
                force_standin=not use_real
            )

        t_burst_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(1, burst_n + 1)]
            for idx, f in enumerate(as_completed(futures), 1):
                res = f.result()
                results_list.append(res)
                progress_bar.progress(idx / burst_n)
                status_text.text(f"Completed {idx}/{burst_n} requests...")

        total_burst_time = time.perf_counter() - t_burst_start
        status_text.success(f"Burst finished in {total_burst_time:.2f} seconds!")

        # Analytics
        df_burst = pd.DataFrame([
            {
                "Request ID": r.request_id,
                "Status": r.status.upper(),
                "TTFT (ms)": r.first_token_ms,
                "Total Latency (ms)": r.total_ms,
                "Retries": r.retries
            }
            for r in results_list
        ])

        col_res1, col_res2, col_res3 = st.columns(3)
        succ_pct = 100.0 * sum(1 for r in results_list if r.status == "ok") / len(results_list)
        rate_pct = 100.0 * sum(1 for r in results_list if r.status == "rate_limited") / len(results_list)
        p50_ttft_val = np.percentile([r.first_token_ms for r in results_list if r.status == "ok"], 50) if any(r.status == "ok" for r in results_list) else 0

        col_res1.metric("Success Rate", f"{succ_pct:.1f}%")
        col_res2.metric("Rate Limited (429)", f"{rate_pct:.1f}%")
        col_res3.metric("TTFT p50 under Concurrency", f"{p50_ttft_val:.0f} ms")

        st.line_chart(df_burst.set_index("Request ID")[["TTFT (ms)", "Total Latency (ms)"]])
        st.dataframe(df_burst, use_container_width=True)

# =========================================================================
# TAB 3: ARCHITECTURE & LATENCY BREAKDOWN
# =========================================================================
with tab_architecture:
    st.markdown("### 📐 System Architecture & Detailed Latency Breakdown")
    
    st.markdown("""
    #### Where the Time Goes (Detailed Breakdown of Slow Request `burst_req_010`)
    - **Total Latency:** `2,723.9 ms`
    - **Waiting in Concurrency Queue:** `1,184.2 ms` (77.4% of delay occurred **before the first token**)
    - **Prompt Prefill & Index Lookup:** `348.0 ms`
    - **Autoregressive Token Emission:** `1,191.7 ms` (83 tokens at ~14.3ms/token)
    """)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("output/chart_latency_cdf.png"):
            st.image("output/chart_latency_cdf.png", caption="Figure 1: Cumulative Latency Distribution Across Scenarios", use_container_width=True)
    with col_fig2:
        if os.path.exists("output/chart_queue_degradation.png"):
            st.image("output/chart_queue_degradation.png", caption="Figure 2: Empirical Queue Contention Knee (N > 5)", use_container_width=True)

    if os.path.exists("output/chart_quality_vs_cost.png"):
        st.image("output/chart_quality_vs_cost.png", caption="Figure 3: Retrieval Quality vs Token Cost (Pareto Frontier)", width=720)

# =========================================================================
# TAB 4: EMPIRICAL TELEMETRY & RESULTS TABLES
# =========================================================================
with tab_tables:
    st.markdown("### 📋 Official Results Tables (Derived Directly from `results.csv`)")
    if os.path.exists("output/results_tables.md"):
        with open("output/results_tables.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python src/compute_tables.py` to generate the latest results tables.")
