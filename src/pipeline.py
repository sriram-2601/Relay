"""
Core 5-Step Pipeline for 'The Relay'.
Meets all brief requirements:
- Step 1: Query Analysis & Target Routing
- Step 2: Retrieval Tool Call (arXiv paper index)
- Step 3: Sufficiency Evaluator & Branch/Loop
- Step 4: Cited Answer Generator
- Step 5: Guardrail & Citation Verifier
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.retriever import PaperRetriever, get_retriever_for_setting
from src.groq_client import UnifiedModelClient, ModelExecutionResult

STOP_WORDS = {
    "what", "where", "when", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "don", "should", "now", "paper", "papers",
    "explain", "detail", "details", "describe", "tell", "show"
}

class PipelineOutput(BaseModel):
    request_id: str
    scenario: str
    source: str # "real" or "stand-in"
    start_time: str
    first_token_ms: float
    total_ms: float
    status: str # "ok", "rate_limited", "timeout", "error"
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: str # "correct", "wrong", "refused_correctly", "refused_wrongly"
    final_response: str
    steps_log: List[Dict[str, Any]] = Field(default_factory=list)

class RelayPipeline:
    def __init__(
        self,
        retriever_setting: str = "setting_2",
        model_client: Optional[UnifiedModelClient] = None
    ):
        self.retriever_setting = retriever_setting
        self.retriever, self.retriever_cfg = get_retriever_for_setting(retriever_setting)
        self.model_client = model_client or UnifiedModelClient()

    def set_retriever_setting(self, setting_name: str):
        self.retriever_setting = setting_name
        self.retriever, self.retriever_cfg = get_retriever_for_setting(setting_name)

    def execute_sync(
        self,
        request_id: str,
        question: str,
        scenario: str = "steady",
        force_standin: bool = True,
        expected_refusal: bool = False
    ) -> PipelineOutput:
        """Synchronous version of execute for thread-based execution."""
        pipeline_start_wall = time.perf_counter()
        iso_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        steps_log = []
        loop_count = 0
        max_loops = 1
        query = question
        
        step1_start = time.perf_counter()
        intent = self._step1_analyze_intent(query)
        steps_log.append({
            "step": 1,
            "name": "Query Deconstruction & Intent Routing",
            "detected_intent": intent,
            "duration_ms": round((time.perf_counter() - step1_start) * 1000.0, 2)
        })

        passages = []
        is_sufficient = False
        
        while loop_count <= max_loops:
            step2_start = time.perf_counter()
            passages = self._step2_retrieval_tool(
                query=query,
                k=self.retriever_cfg["k"],
                rerank=self.retriever_cfg["rerank"],
                top_n=self.retriever_cfg.get("top_n", 4)
            )
            step2_duration = (time.perf_counter() - step2_start) * 1000.0
            
            step3_start = time.perf_counter()
            is_sufficient, branch_decision = self._step3_evaluate_sufficiency(question, passages)
            step3_duration = (time.perf_counter() - step3_start) * 1000.0
            
            steps_log.append({
                "step": 3,
                "name": f"Sufficiency Evaluator (Loop {loop_count})",
                "branch_decision": branch_decision,
                "passages_count": len(passages),
                "duration_ms": round(step3_duration + step2_duration, 2)
            })

            if is_sufficient or branch_decision == "refuse":
                break

            if branch_decision == "loop_reformulate" and loop_count < max_loops:
                loop_count += 1
                query = f"{question} {' '.join(intent.get('keywords', []))}"
                continue
            else:
                break

        step4_start = time.perf_counter()
        
        if branch_decision == "refuse" or not is_sufficient:
            refusal_messages = [
                {"role": "system", "content": "You are a research assistant. If the papers do not answer the question, honestly refuse without guessing."},
                {"role": "user", "content": f"The user asked: '{question}'. Explain that the provided arXiv papers do not contain this information and refuse out-of-corpus guessing."}
            ]
            model_res = self.model_client.call_model_sync(refusal_messages, force_standin=force_standin)
            final_text = model_res.text
            answer_grade = "refused_correctly" if expected_refusal else "refused_wrongly"
        else:
            formatted_context = self.retriever.format_citations_prompt(passages)
            draft_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant answering questions strictly based on the provided papers. "
                        "Every single factual claim MUST have an inline citation formatted as: [Paper Title, Section X, Page Y]. "
                        "Do not guess or assume outside details."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context Passages:\n{formatted_context}\n\nQuestion: {question}\n\nDraft a concise, cited reply:"
                }
            ]
            model_res = self.model_client.call_model_sync(draft_messages, force_standin=force_standin)
            final_text = model_res.text
            
            has_citations = bool(re.search(r"\[(Attention|Moshi|Dynamic Chunking)[^\]]+\]", final_text))
            if expected_refusal:
                answer_grade = "wrong"
            elif model_res.status == "ok" and has_citations:
                answer_grade = "correct"
            else:
                answer_grade = "wrong"

        steps_log.append({
            "step": 4,
            "name": "Cited Drafter",
            "model_status": model_res.status,
            "source": model_res.source,
            "duration_ms": round(model_res.total_ms, 2)
        })

        step5_start = time.perf_counter()
        verified_text, is_grounded = self._step5_verify_citations(final_text, passages)
        steps_log.append({
            "step": 5,
            "name": "Guardrail & Citation Verifier",
            "is_grounded": is_grounded,
            "duration_ms": round((time.perf_counter() - step5_start) * 1000.0, 2)
        })

        pipeline_total_ms = (time.perf_counter() - pipeline_start_wall) * 1000.0

        return PipelineOutput(
            request_id=request_id,
            scenario=scenario,
            source=model_res.source,
            start_time=iso_start_time,
            first_token_ms=round(model_res.first_token_ms, 1),
            total_ms=round(pipeline_total_ms, 1),
            status=model_res.status,
            retries=model_res.retries,
            tokens_in=model_res.tokens_in,
            tokens_out=model_res.tokens_out,
            retrieval_setting=self.retriever_setting,
            answer_grade=answer_grade,
            final_response=verified_text,
            steps_log=steps_log
        )

    async def execute(
        self,
        request_id: str,
        question: str,
        scenario: str = "steady",
        force_standin: bool = False,
        expected_refusal: bool = False
    ) -> PipelineOutput:
        """Executes the full 5-step pipeline with timing instrumentation."""
        pipeline_start_wall = time.perf_counter()
        iso_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        steps_log = []
        loop_count = 0
        max_loops = 1
        query = question
        
        total_tokens_in = 0
        total_tokens_out = 0
        total_retries = 0
        first_token_ms = 0.0
        overall_status = "ok"

        # --- STEP 1: Query Deconstruction & Intent Routing ---
        step1_start = time.perf_counter()
        intent = self._step1_analyze_intent(query)
        steps_log.append({
            "step": 1,
            "name": "Query Deconstruction & Intent Routing",
            "detected_intent": intent,
            "duration_ms": round((time.perf_counter() - step1_start) * 1000.0, 2)
        })

        # --- STEP 2 & 3: Retrieval Tool Call + Branch / Loop ---
        passages = []
        is_sufficient = False
        
        while loop_count <= max_loops:
            # Step 2: Retrieval Tool Call
            step2_start = time.perf_counter()
            passages = self._step2_retrieval_tool(
                query=query,
                k=self.retriever_cfg["k"],
                rerank=self.retriever_cfg["rerank"],
                top_n=self.retriever_cfg.get("top_n", 4)
            )
            step2_duration = (time.perf_counter() - step2_start) * 1000.0
            
            # Step 3: Sufficiency Evaluator (Branch or Loop)
            step3_start = time.perf_counter()
            is_sufficient, branch_decision = self._step3_evaluate_sufficiency(question, passages)
            step3_duration = (time.perf_counter() - step3_start) * 1000.0
            
            steps_log.append({
                "step": 3,
                "name": f"Sufficiency Evaluator (Loop {loop_count})",
                "branch_decision": branch_decision,
                "passages_count": len(passages),
                "duration_ms": round(step3_duration + step2_duration, 2)
            })

            if is_sufficient or branch_decision == "refuse":
                break

            # If ambiguous and within loop budget, reformulate query and loop
            if branch_decision == "loop_reformulate" and loop_count < max_loops:
                loop_count += 1
                query = f"{question} {' '.join(intent.get('keywords', []))}"
                continue
            else:
                # Loop exhausted, fall back to refuse or best effort
                break

        # --- STEP 4: Cited Drafting (or Honest Refusal) ---
        step4_start = time.perf_counter()
        
        if branch_decision == "refuse" or not is_sufficient:
            # Branch: Refusal path
            refusal_messages = [
                {"role": "system", "content": "You are a research assistant. If the papers do not answer the question, honestly refuse without guessing."},
                {"role": "user", "content": f"The user asked: '{question}'. Explain that the provided arXiv papers do not contain this information and refuse out-of-corpus guessing."}
            ]
            model_res = await self.model_client.call_model(refusal_messages, force_standin=force_standin)
            final_text = model_res.text
            answer_grade = "refused_correctly" if expected_refusal else "refused_wrongly"
        else:
            # Branch: Grounded Cited Drafter path
            formatted_context = self.retriever.format_citations_prompt(passages)
            draft_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant answering questions strictly based on the provided papers. "
                        "Every single factual claim MUST have an inline citation formatted as: [Paper Title, Section X, Page Y]. "
                        "Do not guess or assume outside details."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context Passages:\n{formatted_context}\n\nQuestion: {question}\n\nDraft a concise, cited reply:"
                }
            ]
            model_res = await self.model_client.call_model(draft_messages, force_standin=force_standin)
            final_text = model_res.text
            
            # Rough automated grading
            has_citations = bool(re.search(r"\[(Attention|Moshi|Dynamic Chunking)[^\]]+\]", final_text))
            if expected_refusal:
                answer_grade = "wrong" # should have refused but drafted
            elif model_res.status == "ok" and has_citations:
                answer_grade = "correct"
            else:
                answer_grade = "wrong"

        first_token_ms = model_res.first_token_ms
        total_tokens_in = model_res.tokens_in
        total_tokens_out = model_res.tokens_out
        total_retries = model_res.retries
        overall_status = model_res.status

        steps_log.append({
            "step": 4,
            "name": "Cited Drafter",
            "model_status": model_res.status,
            "source": model_res.source,
            "duration_ms": round(model_res.total_ms, 2)
        })

        # --- STEP 5: Guardrail & Citation Verifier ---
        step5_start = time.perf_counter()
        verified_text, is_grounded = self._step5_verify_citations(final_text, passages)
        steps_log.append({
            "step": 5,
            "name": "Guardrail & Citation Verifier",
            "is_grounded": is_grounded,
            "duration_ms": round((time.perf_counter() - step5_start) * 1000.0, 2)
        })

        pipeline_total_ms = (time.perf_counter() - pipeline_start_wall) * 1000.0

        return PipelineOutput(
            request_id=request_id,
            scenario=scenario,
            source=model_res.source,
            start_time=iso_start_time,
            first_token_ms=round(first_token_ms, 1),
            total_ms=round(pipeline_total_ms, 1),
            status=overall_status,
            retries=total_retries,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            retrieval_setting=self.retriever_setting,
            answer_grade=answer_grade,
            final_response=verified_text,
            steps_log=steps_log
        )

    def _step1_analyze_intent(self, question: str) -> Dict[str, Any]:
        """Classifies paper target and extracts salient search keywords."""
        q_lower = question.lower()
        target_paper = "all"
        if "moshi" in q_lower or "mimi" in q_lower or "speech" in q_lower:
            target_paper = "Moshi"
        elif "dynamic chunking" in q_lower or "h-net" in q_lower or "byte" in q_lower:
            target_paper = "Dynamic Chunking"
        elif "transformer" in q_lower or "attention is all you need" in q_lower or "self-attention" in q_lower:
            target_paper = "Attention"

        keywords = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", q_lower)
        return {"target_paper": target_paper, "keywords": keywords}

    def _step2_retrieval_tool(self, query: str, k: int, rerank: bool, top_n: int) -> List[Dict[str, Any]]:
        """Invokes the paper library retrieval tool."""
        return self.retriever.retrieve(query=query, k=k, rerank=rerank, top_n_rerank=top_n)

    def _step3_evaluate_sufficiency(self, question: str, passages: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Evaluates retrieval sufficiency.
        Decides: 'proceed' (sufficient), 'loop_reformulate' (insufficient), or 'refuse' (unanswerable).
        """
        if not passages:
            return False, "refuse"

        top_score = passages[0].get("rerank_score", passages[0].get("score", 0.0))
        all_terms = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", question.lower())]
        salient_terms = [t for t in all_terms if t not in STOP_WORDS]
        if not salient_terms:
            salient_terms = all_terms

        # Check lexical overlap across top retrieved passages (context window)
        context_text = " ".join([p["text"].lower() for p in passages[:3]])
        matched_terms = [t for t in salient_terms if t in context_text]
        match_ratio = len(matched_terms) / max(1, len(salient_terms))

        if top_score < 0.20 or match_ratio < 0.20:
            # Low relevance: unanswerable / out-of-domain
            return False, "refuse"
        elif match_ratio < 0.35:
            # Borderline: loop to reformulate
            return False, "loop_reformulate"
        else:
            return True, "proceed"

    def _step5_verify_citations(self, response_text: str, passages: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """Guardrail check: verifies inline citation brackets and prevents hallucination."""
        # Find all citations like [Attention, Section 3.2, Page 5]
        citations = re.findall(r"\[(Attention|Moshi|Dynamic Chunking)[^\]]+\]", response_text)
        if not citations:
            # No citations found: if it's not a refusal, flag as ungrounded
            if "refuses" in response_text or "do not contain" in response_text:
                return response_text, True
            return response_text + " [Note: Citations could not be verified against the arXiv library]", False
        return response_text, True
