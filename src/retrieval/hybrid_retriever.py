from __future__ import annotations

from pathlib import Path

from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.schemas import RetrievalResult
from src.retrieval.weighted_fusion import weighted_score_fusion

class HybridRetriever:
    """
    Offline hybrid retriever combining BM25 and dense FAISS retrieval.

    Uses:
    - BM25 for lexical/entity precision
    - Dense retrieval for semantic similarity
    - Reciprocal Rank Fusion for robust ranking
    """

    def __init__(
            self,
            bm25_retriever: BM25Retriever,
            dense_retriever: DenseRetriever,
            *,
            fusion_method: str = "rrf",
            rrf_k: int = 60,
            bm25_candidate_k: int = 30,
            dense_candidate_k: int = 30,
            bm25_weight: float = 0.7,
            dense_weight: float = 0.3,
    ):
        if bm25_candidate_k <= 0:
            raise ValueError("bm25_candidate_k must be positive.")
        if dense_candidate_k <= 0:
            raise ValueError("dense_candidate_k must be positive.")
        if fusion_method not in {"rrf", "weighted"}:
            raise ValueError("fusion_method must be one of: rrf, weighted")

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.bm25_candidate_k = bm25_candidate_k
        self.dense_candidate_k = dense_candidate_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=self.bm25_candidate_k,
        )

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=self.dense_candidate_k,
        )

        if self.fusion_method == "rrf":
            return reciprocal_rank_fusion(
                [bm25_results, dense_results],
                rrf_k=self.rrf_k,
                final_top_k=top_k,
            )

        return weighted_score_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            final_top_k=top_k,
        )

    @classmethod
    def from_indexes(
            cls,
            *,
            bm25_index_dir: str | Path,
            dense_index_dir: str | Path,
            device: str = "cpu",
            fusion_method: str = "rrf",
            rrf_k: int = 60,
            bm25_candidate_k: int = 30,
            dense_candidate_k: int = 30,
            bm25_weight: float = 0.7,
            dense_weight: float = 0.3,
    ) -> "HybridRetriever":
        bm25 = BM25Retriever.load(bm25_index_dir)
        dense = DenseRetriever.load(dense_index_dir, device=device)

        return cls(
            bm25_retriever=bm25,
            dense_retriever=dense,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            bm25_candidate_k=bm25_candidate_k,
            dense_candidate_k=dense_candidate_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query hybrid BM25 + dense retriever.")
    parser.add_argument("--bm25-index-dir", required=True)
    parser.add_argument("--dense-index-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-candidate-k", type=int, default=30)
    parser.add_argument("--dense-candidate-k", type=int, default=30)
    parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument("--dense-weight", type=float, default=0.3)

    args = parser.parse_args()

    retriever = HybridRetriever.from_indexes(
        bm25_index_dir=args.bm25_index_dir,
        dense_index_dir=args.dense_index_dir,
        device=args.device,
        fusion_method=args.fusion_method,
        rrf_k=args.rrf_k,
        bm25_candidate_k=args.bm25_candidate_k,
        dense_candidate_k=args.dense_candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    for result in results:
        print(
            f"\nRank {result.rank} | {result.chunk_id} | "
            f"rrf_score={result.score:.6f} | methods={result.metadata.get('contributing_methods')}"
        )
        print(f"Section: {result.section}")
        print(result.text[:600])