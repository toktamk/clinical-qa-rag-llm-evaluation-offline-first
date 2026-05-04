from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


RetrievalMethod = Literal["bm25", "dense", "hybrid", "reranker"]


@dataclass(frozen=True)
class RetrievalResult:
    """
    Standard retrieval result returned by all retrievers.

    This object is intentionally shared across BM25, dense, hybrid and reranking
    pipelines so downstream generation, citation validation and evaluation code
    can use one consistent interface.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    rank: int
    retrieval_method: RetrievalMethod
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalQuery:
    """
    Input query object.

    Keeping this separate from a raw string makes it easier to later add filters
    such as section filters, document-type filters, or clinical-risk filters.
    """

    query: str
    top_k: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedChunk:
    """
    Internal representation of a chunk used by retrievers.

    This normalises the chunk records loaded from chunks.jsonl and prevents
    retrievers from depending directly on raw JSON structure.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    token_count: int
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, chunk: dict[str, Any]) -> "IndexedChunk":
        return cls(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            title=chunk.get("title", ""),
            section=chunk.get("section", ""),
            text=chunk["text"],
            token_count=int(chunk.get("token_count", 0)),
            source_path=chunk.get("source_path", ""),
            metadata=chunk.get("metadata", {}),
        )

    def to_retrieval_result(
        self,
        *,
        score: float,
        rank: int,
        retrieval_method: RetrievalMethod,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            title=self.title,
            section=self.section,
            text=self.text,
            score=float(score),
            rank=int(rank),
            retrieval_method=retrieval_method,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalRun:
    """
    Optional container for a retrieval run.

    Useful later when saving retrieval outputs for evaluation and debugging.
    """

    query: str
    method: RetrievalMethod
    top_k: int
    results: list[RetrievalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "top_k": self.top_k,
            "results": [result.to_dict() for result in self.results],
        }