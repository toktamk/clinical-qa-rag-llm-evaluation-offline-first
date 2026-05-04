from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.schemas import IndexedChunk, RetrievalResult
from src.retrieval.text_normalisation import chunk_text_for_indexing, tokenize_for_bm25, normalise_query


class BM25Retriever:
    """
    Offline BM25 retriever over chunks.jsonl.

    This retriever is the sparse lexical baseline for the clinical RAG system.
    It is useful for exact matches such as:
    - therapy names
    - condition names
    - section names
    - numerical thresholds
    - monitoring intervals
    """

    def __init__(self, chunks: list[dict[str, Any]]):
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")

        self.chunks: list[IndexedChunk] = [IndexedChunk.from_dict(chunk) for chunk in chunks]
        self.index_texts: list[str] = [chunk_text_for_indexing(chunk) for chunk in chunks]
        self.tokenised_corpus: list[list[str]] = [
            tokenize_for_bm25(text) for text in self.index_texts
        ]

        self.bm25 = BM25Okapi(self.tokenised_corpus)

    @classmethod
    def from_jsonl(cls, chunks_path: str | Path) -> "BM25Retriever":
        chunks = load_jsonl(chunks_path)
        return cls(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query = normalise_query(query)
        query_tokens = tokenize_for_bm25(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                chunk.to_retrieval_result(
                    score=float(scores[idx]),
                    rank=rank,
                    retrieval_method="bm25",
                )
            )

        return results

    def save(self, index_dir: str | Path) -> None:
        """
        Persist BM25 retriever state.

        rank_bm25 indexes are lightweight, so pickle is acceptable here.
        For production-critical systems, store corpus + config and rebuild.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with (index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(
                {
                    "chunks": [chunk.to_dict() for chunk in self.chunks],
                    "index_texts": self.index_texts,
                    "tokenised_corpus": self.tokenised_corpus,
                    "bm25": self.bm25,
                },
                f,
            )

    @classmethod
    def load(cls, index_dir: str | Path) -> "BM25Retriever":
        index_dir = Path(index_dir)
        path = index_dir / "bm25.pkl"

        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")

        with path.open("rb") as f:
            state = pickle.load(f)

        obj = cls.__new__(cls)
        obj.chunks = [IndexedChunk.from_dict(chunk) for chunk in state["chunks"]]
        obj.index_texts = state["index_texts"]
        obj.tokenised_corpus = state["tokenised_corpus"]
        obj.bm25 = state["bm25"]

        return obj


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

    return records


def build_bm25_index(
    chunks_path: str | Path,
    index_dir: str | Path | None = None,
) -> BM25Retriever:
    """
    Build a BM25 retriever from chunks.jsonl.

    Optionally saves the index to disk.
    """
    retriever = BM25Retriever.from_jsonl(chunks_path)

    if index_dir is not None:
        retriever.save(index_dir)

    return retriever


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and query a BM25 index.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--index-dir", default=None, help="Optional output index directory")
    parser.add_argument("--query", default=None, help="Optional test query")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    bm25 = build_bm25_index(args.chunks, args.index_dir)

    print(f"Loaded {len(bm25.chunks)} chunks")

    if args.index_dir:
        print(f"Saved BM25 index to {args.index_dir}")

    if args.query:
        results = bm25.retrieve(args.query, top_k=args.top_k)
        for result in results:
            print(
                f"\nRank {result.rank} | {result.chunk_id} | score={result.score:.4f}"
            )
            print(f"Section: {result.section}")
            print(result.text[:500])