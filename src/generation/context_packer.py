from __future__ import annotations

from typing import Any

from src.generation.schemas import EvidenceChunk, PackedContext


def pack_context(
    question: str,
    retrieved_results: list[Any],
    *,
    max_chunks: int = 5,
    max_chars_per_chunk: int = 1600,
) -> PackedContext:
    """
    Convert retrieved chunks into a citation-ready context block.

    The resulting context explicitly exposes chunk IDs so the generator can cite
    evidence using machine-checkable references.

    Args:
        question: User or evaluation question.
        retrieved_results: RetrievalResult objects or EvidenceChunk/dict records.
        max_chunks: Maximum number of chunks to include.
        max_chars_per_chunk: Character cap per chunk to avoid oversized prompts.

    Returns:
        PackedContext with formatted context_text and used_chunk_ids.
    """
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")

    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be positive.")

    evidence_chunks = [
        to_evidence_chunk(result)
        for result in retrieved_results[:max_chunks]
    ]

    blocks: list[str] = []
    used_chunk_ids: list[str] = []

    for chunk in evidence_chunks:
        used_chunk_ids.append(chunk.chunk_id)

        chunk_text = truncate_text(chunk.text, max_chars=max_chars_per_chunk)

        blocks.append(
            "\n".join(
                [
                    f"[CHUNK_ID: {chunk.chunk_id}]",
                    f"Document: {chunk.doc_id}",
                    f"Title: {chunk.title}",
                    f"Section: {chunk.section}",
                    f"Rank: {chunk.rank}",
                    f"Retrieval method: {chunk.retrieval_method}",
                    "Text:",
                    chunk_text,
                ]
            )
        )

    context_text = "\n\n---\n\n".join(blocks)

    return PackedContext(
        question=question,
        chunks=evidence_chunks,
        context_text=context_text,
        used_chunk_ids=used_chunk_ids,
    )


def to_evidence_chunk(result: Any) -> EvidenceChunk:
    """
    Convert supported retrieval result formats into EvidenceChunk.
    """
    if isinstance(result, EvidenceChunk):
        return result

    if isinstance(result, dict):
        return EvidenceChunk.from_dict(result)

    required_attrs = ["chunk_id", "doc_id", "title", "section", "text"]

    if all(hasattr(result, attr) for attr in required_attrs):
        return EvidenceChunk.from_retrieval_result(result)

    raise TypeError(
        "Unsupported retrieval result type. Expected EvidenceChunk, dict, "
        "or object with chunk_id/doc_id/title/section/text attributes."
    )


def truncate_text(text: str, *, max_chars: int) -> str:
    """
    Truncate long chunk text safely.

    Adds an explicit truncation marker so downstream debugging is transparent.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n[TRUNCATED]"