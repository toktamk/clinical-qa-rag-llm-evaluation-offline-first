from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.schemas import IndexedChunk, RetrievalResult
from src.retrieval.text_normalisation import chunk_text_for_indexing


class DenseRetriever:
    """
    Offline dense retriever using sentence-transformers + FAISS.

    Uses cosine similarity via:
    - sentence embeddings
    - L2 normalisation
    - FAISS IndexFlatIP
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk.")

        self.chunks: list[IndexedChunk] = [IndexedChunk.from_dict(chunk) for chunk in chunks]
        self.raw_chunks = chunks
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.batch_size = batch_size

        self.model = SentenceTransformer(embedding_model_name, device=device)
        self.index: faiss.Index | None = None
        self.embeddings: np.ndarray | None = None

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: str | Path,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> "DenseRetriever":
        chunks = load_jsonl(chunks_path)
        return cls(
            chunks=chunks,
            embedding_model_name=embedding_model_name,
            device=device,
            batch_size=batch_size,
        )

    def build_index(self) -> None:
        texts = [
            f"Represent this sentence for searching relevant passages: {chunk_text_for_indexing(chunk)}"
            for chunk in self.raw_chunks
        ]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.embeddings = embeddings
        self.index = index

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index is None:
            raise RuntimeError("Dense index has not been built or loaded.")

        query_text = f"Represent this sentence for searching relevant passages: {query}"

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results: list[RetrievalResult] = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue

            chunk = self.chunks[int(idx)]
            results.append(
                chunk.to_retrieval_result(
                    score=float(score),
                    rank=rank,
                    retrieval_method="dense",
                )
            )

        return results

    def save(self, index_dir: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save dense retriever before build_index().")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "index.faiss"))

        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "raw_chunks": self.raw_chunks,
        }

        with (index_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        device: str = "cpu",
    ) -> "DenseRetriever":
        index_dir = Path(index_dir)

        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dense metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        obj = cls(
            chunks=metadata["raw_chunks"],
            embedding_model_name=metadata["embedding_model_name"],
            device=device,
            batch_size=metadata.get("batch_size", 32),
        )

        obj.index = faiss.read_index(str(index_path))
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


def build_dense_index(
    chunks_path: str | Path,
    index_dir: str | Path | None = None,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
    batch_size: int = 32,
) -> DenseRetriever:
    retriever = DenseRetriever.from_jsonl(
        chunks_path=chunks_path,
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )

    retriever.build_index()

    if index_dir is not None:
        retriever.save(index_dir)

    return retriever


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and query a dense FAISS index.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--index-dir", default=None, help="Optional output index directory")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformers embedding model name",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query", default=None, help="Optional test query")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    dense = build_dense_index(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"Loaded {len(dense.chunks)} chunks")

    if args.index_dir:
        print(f"Saved dense FAISS index to {args.index_dir}")

    if args.query:
        results = dense.retrieve(args.query, top_k=args.top_k)

        for result in results:
            print(
                f"\nRank {result.rank} | {result.chunk_id} | score={result.score:.4f}"
            )
            print(f"Section: {result.section}")
            print(result.text[:500])