from __future__ import annotations

from src.retrieval.schemas import RetrievalResult


def min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalise scores into [0, 1].

    If all scores are identical, assign 1.0 to all non-empty scores.
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {key: 1.0 for key in scores}

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def weighted_score_fusion(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    *,
    bm25_weight: float = 0.7,
    dense_weight: float = 0.3,
    final_top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Fuse BM25 and dense retrieval using weighted normalised scores.

    hybrid_score = bm25_weight * normalised_bm25_score
                 + dense_weight * normalised_dense_score
    """
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive.")

    if bm25_weight < 0 or dense_weight < 0:
        raise ValueError("Fusion weights must be non-negative.")

    if bm25_weight + dense_weight == 0:
        raise ValueError("At least one fusion weight must be positive.")

    # Normalise weights in case user passes values that do not sum to 1.
    total_weight = bm25_weight + dense_weight
    bm25_weight = bm25_weight / total_weight
    dense_weight = dense_weight / total_weight

    bm25_scores = {result.chunk_id: result.score for result in bm25_results}
    dense_scores = {result.chunk_id: result.score for result in dense_results}

    bm25_norm = min_max_normalise(bm25_scores)
    dense_norm = min_max_normalise(dense_scores)

    all_chunk_ids = set(bm25_norm) | set(dense_norm)

    best_result_by_chunk_id: dict[str, RetrievalResult] = {}
    contributing_methods: dict[str, list[str]] = {}

    for result in bm25_results + dense_results:
        chunk_id = result.chunk_id

        if chunk_id not in best_result_by_chunk_id:
            best_result_by_chunk_id[chunk_id] = result
        elif result.score > best_result_by_chunk_id[chunk_id].score:
            best_result_by_chunk_id[chunk_id] = result

        contributing_methods.setdefault(chunk_id, [])
        contributing_methods[chunk_id].append(result.retrieval_method)

    hybrid_scores = {}

    for chunk_id in all_chunk_ids:
        hybrid_scores[chunk_id] = (
            bm25_weight * bm25_norm.get(chunk_id, 0.0)
            + dense_weight * dense_norm.get(chunk_id, 0.0)
        )

    ranked_chunk_ids = sorted(
        hybrid_scores.keys(),
        key=lambda chunk_id: hybrid_scores[chunk_id],
        reverse=True,
    )[:final_top_k]

    fused_results: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        base = best_result_by_chunk_id[chunk_id]

        metadata = dict(base.metadata)
        metadata["fusion"] = "weighted_score"
        metadata["hybrid_score"] = hybrid_scores[chunk_id]
        metadata["normalised_bm25_score"] = bm25_norm.get(chunk_id, 0.0)
        metadata["normalised_dense_score"] = dense_norm.get(chunk_id, 0.0)
        metadata["bm25_weight"] = bm25_weight
        metadata["dense_weight"] = dense_weight
        metadata["contributing_methods"] = sorted(set(contributing_methods[chunk_id]))

        fused_results.append(
            RetrievalResult(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                title=base.title,
                section=base.section,
                text=base.text,
                score=float(hybrid_scores[chunk_id]),
                rank=rank,
                retrieval_method="hybrid",
                metadata=metadata,
            )
        )

    return fused_results