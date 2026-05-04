from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


VerificationStatus = Literal["passed", "failed", "warning"]
AbstentionReason = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "ambiguous_evidence",
    "out_of_scope",
    "safety_boundary",
]


@dataclass(frozen=True)
class EvidenceChunk:
    """
    Evidence chunk passed into the generation layer.

    This decouples generation from the retrieval implementation while preserving
    citation-critical metadata.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float | None = None
    rank: int | None = None
    retrieval_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_result(cls, result: Any) -> "EvidenceChunk":
        return cls(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            title=result.title,
            section=result.section,
            text=result.text,
            score=getattr(result, "score", None),
            rank=getattr(result, "rank", None),
            retrieval_method=getattr(result, "retrieval_method", None),
            metadata=getattr(result, "metadata", {}) or {},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceChunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            section=data.get("section", ""),
            text=data["text"],
            score=data.get("score"),
            rank=data.get("rank"),
            retrieval_method=data.get("retrieval_method"),
            metadata=data.get("metadata", {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackedContext:
    """
    Context block passed to the generator.

    used_chunk_ids is the authoritative list of chunks the model was allowed to cite.
    """

    question: str
    chunks: list[EvidenceChunk]
    context_text: str
    used_chunk_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "context_text": self.context_text,
            "used_chunk_ids": self.used_chunk_ids,
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    """
    Raw generated answer plus parsed citation and abstention metadata.
    """

    question: str
    answer: str
    citations: list[str]
    abstained: bool
    abstention_reason: AbstentionReason | None
    used_chunk_ids: list[str]
    raw_model_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    """
    Deterministic verification result for generated answers.

    This is not a semantic entailment verifier yet. It checks citation validity,
    abstention consistency, and basic safety/format issues.
    """

    verification_status: VerificationStatus
    unsupported_claims: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    safety_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verification_status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RAGResponse:
    """
    Final response object returned by the RAG pipeline.
    """

    generated_answer: GeneratedAnswer
    verification: VerificationResult
    retrieved_chunks: list[EvidenceChunk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_answer": self.generated_answer.to_dict(),
            "verification": self.verification.to_dict(),
            "retrieved_chunks": [chunk.to_dict() for chunk in self.retrieved_chunks],
        }