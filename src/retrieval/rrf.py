from __future__ import annotations

from collections import defaultdict

from src.retrieval.schemas import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    *,
    rrf_k: int = 60,
    final_top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Combine ranked retrieval results using Reciprocal Rank Fusion.

    RRF score:
        sum(1 / (rrf_k + rank))

    Args:
        result_lists: List of ranked result lists from different retrievers.
        rrf_k: Stabilisation constant. Common default: 60.
        final_top_k: Number of fused results to return.

    Returns:
        Fused retrieval results sorted by RRF score.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive.")

    scores: dict[str, float] = defaultdict(float)
    best_result_by_chunk_id: dict[str, RetrievalResult] = {}
    contributing_methods: dict[str, list[str]] = defaultdict(list)

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk_id
            scores[chunk_id] += 1.0 / (rrf_k + result.rank)

            if chunk_id not in best_result_by_chunk_id:
                best_result_by_chunk_id[chunk_id] = result
            elif result.score > best_result_by_chunk_id[chunk_id].score:
                best_result_by_chunk_id[chunk_id] = result

            contributing_methods[chunk_id].append(result.retrieval_method)

    ranked_chunk_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )[:final_top_k]

    fused_results: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        base = best_result_by_chunk_id[chunk_id]
        metadata = dict(base.metadata)
        metadata["rrf_score"] = scores[chunk_id]
        metadata["contributing_methods"] = sorted(set(contributing_methods[chunk_id]))

        fused_results.append(
            RetrievalResult(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                title=base.title,
                section=base.section,
                text=base.text,
                score=float(scores[chunk_id]),
                rank=rank,
                retrieval_method="hybrid",
                metadata=metadata,
            )
        )

    return fused_results