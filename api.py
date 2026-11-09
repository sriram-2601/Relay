"""
The Relay — Production REST API (FastAPI)
Exposes the 5-step research assistant pipeline as standard JSON endpoints.
Run with: uvicorn api:app --reload --port 8000
"""

import os
import sys
import time
from typing import Dict, Any, Optional
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import RelayPipeline
from src.standin_service import CalibratedStandInModel
from src.groq_client import UnifiedModelClient

app = FastAPI(
    title="The Relay — Grounded Research Assistant API",
    description="Production REST API for cited research Q&A with rate-limiting and honest refusal gates.",
    version="1.0.0"
)

# Global pipeline instance
standin = CalibratedStandInModel()
client = UnifiedModelClient(standin_model=standin)
pipeline = RelayPipeline(retriever_setting="setting_1", model_client=client)

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the dimension d_model in Attention Is All You Need?")
    retrieval_setting: Optional[str] = Field("setting_1", example="setting_1")
    force_standin: Optional[bool] = Field(True, example=True)

class QueryResponse(BaseModel):
    request_id: str
    status: str
    answer_grade: str
    response: str
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    steps_log: list

@app.get("/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "The Relay Research Assistant",
        "corpus": [
            "Attention Is All You Need (arXiv:1706.03762v7)",
            "Moshi: a speech-text foundation model (arXiv:2410.00037v1)",
            "Dynamic Chunking (arXiv:2507.07955v2)"
        ],
        "default_retrieval_setting": pipeline.retriever_setting
    }

@app.post("/v1/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    req_id = f"api_{int(time.time() * 1000)}"
    if req.retrieval_setting != pipeline.retriever_setting:
        pipeline.set_retriever_setting(req.retrieval_setting)

    out = pipeline.execute_sync(
        request_id=req_id,
        question=req.question,
        scenario="api_rest",
        force_standin=req.force_standin
    )

    if out.status == "rate_limited":
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please back off.")

    return QueryResponse(
        request_id=out.request_id,
        status=out.status,
        answer_grade=out.answer_grade,
        response=out.final_response,
        first_token_ms=out.first_token_ms,
        total_ms=out.total_ms,
        tokens_in=out.tokens_in,
        tokens_out=out.tokens_out,
        steps_log=out.steps_log
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

