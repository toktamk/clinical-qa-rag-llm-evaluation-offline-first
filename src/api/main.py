from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.local_llm import LocalLLM, LocalLLMConfig, MockLLM
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

_RETRIEVER_CACHE = {}
_LLM_CACHE = {}

app = FastAPI(
    title="Offline Clinical RAG API",
    description=(
        "Offline-first, evaluation-driven clinical RAG API with retrieval, "
        "citation-grounded generation, verification, and abstention."
    ),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    fusion_method: Literal["rrf", "weighted"] = "weighted"
    generation_mode: Literal["extractive", "llm", "extract_then_rewrite"] = "extractive"

    top_k: int = Field(default=10, ge=1, le=50)
    max_context_chunks: int = Field(default=6, ge=1, le=20)
    max_chars_per_chunk: int = Field(default=1200, ge=100, le=5000)

    bm25_index_dir: str = "data/indexes/bm25"
    dense_index_dir: str = "data/indexes/faiss"

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    max_new_tokens: int = Field(default=256, ge=16, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    do_sample: bool = False
    mock_llm: bool = False

    rrf_k: int = 60
    bm25_candidate_k: int = 50
    dense_candidate_k: int = 50
    bm25_weight: float = 0.7
    dense_weight: float = 0.3


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    fusion_method: Literal["rrf", "weighted"] = "weighted"
    top_k: int = Field(default=10, ge=1, le=50)

    bm25_index_dir: str = "data/indexes/bm25"
    dense_index_dir: str = "data/indexes/faiss"
    device: str = "cpu"

    rrf_k: int = 60
    bm25_candidate_k: int = 50
    dense_candidate_k: int = 50
    bm25_weight: float = 0.7
    dense_weight: float = 0.3


class EvaluateRequest(QueryRequest):
    expected_citations: list[str] = Field(default_factory=list)
    requires_abstention: bool = False

def load_retriever(request):
    key = (
        request.retriever,
        request.fusion_method,
        request.bm25_index_dir,
        request.dense_index_dir,
        request.device,
        request.bm25_weight,
        request.dense_weight,
    )

    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]

    if request.retriever == "bm25":
        retriever = BM25Retriever.load(request.bm25_index_dir)
    elif request.retriever == "dense":
        retriever = DenseRetriever.load(request.dense_index_dir, device=request.device)
    elif request.retriever == "hybrid":
        retriever = HybridRetriever.from_indexes(
            bm25_index_dir=request.bm25_index_dir,
            dense_index_dir=request.dense_index_dir,
            device=request.device,
            fusion_method=request.fusion_method,
            rrf_k=request.rrf_k,
            bm25_candidate_k=request.bm25_candidate_k,
            dense_candidate_k=request.dense_candidate_k,
            bm25_weight=request.bm25_weight,
            dense_weight=request.dense_weight,
        )
    else:
        raise ValueError(f"Unsupported retriever: {request.retriever}")

    _RETRIEVER_CACHE[key] = retriever
    return retriever


def load_llm(request: QueryRequest):
    if request.generation_mode == "extractive" or request.mock_llm:
        return MockLLM()

    return LocalLLM(
        LocalLLMConfig(
            model_name=request.model,
            device=request.device,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )
    )


def serialise_retrieval_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "__dict__"):
        return dict(result.__dict__)

    return dict(result)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "offline-clinical-rag-api",
    }


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    start = time.perf_counter()

    try:
        retriever = load_retriever(request)
        results = retriever.retrieve(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "query": request.query,
        "retriever": request.retriever,
        "fusion_method": request.fusion_method,
        "top_k": request.top_k,
        "latency_ms": round(latency_ms, 2),
        "results": [serialise_retrieval_result(r) for r in results],
    }


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    start = time.perf_counter()

    try:
        retriever = load_retriever(request)
        llm = load_llm(request)

        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            generation_mode=request.generation_mode,
            max_context_chunks=request.max_context_chunks,
            max_chars_per_chunk=request.max_chars_per_chunk,
        )

        response = pipeline.answer(
            question=request.query,
            top_k=request.top_k,
        )

        response_dict = response.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    response_dict["api_latency_ms"] = round(latency_ms, 2)

    return response_dict


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    response = query(request)

    generated = response.get("generated_answer", {})
    verification = response.get("verification", {})

    predicted_citations = set(generated.get("citations", []) or [])
    expected_citations = set(request.expected_citations)

    correct_citations = predicted_citations & expected_citations

    citation_precision = (
        len(correct_citations) / len(predicted_citations)
        if predicted_citations
        else 0.0
    )
    citation_recall = (
        len(correct_citations) / len(expected_citations)
        if expected_citations
        else 0.0
    )

    abstained = bool(generated.get("abstained", False))

    return {
        "query": request.query,
        "generation_mode": request.generation_mode,
        "metrics": {
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "abstention_correct": abstained == request.requires_abstention,
            "verification_status": verification.get("verification_status"),
        },
        "response": response,
    }