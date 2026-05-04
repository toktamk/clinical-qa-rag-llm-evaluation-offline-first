from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.generation.local_llm import LocalLLM, LocalLLMConfig, MockLLM
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_retriever(args):
    if args.retriever == "bm25":
        if not args.bm25_index_dir:
            raise ValueError("--bm25-index-dir is required for --retriever bm25")

        return BM25Retriever.load(args.bm25_index_dir)

    if args.retriever == "dense":
        if not args.dense_index_dir:
            raise ValueError("--dense-index-dir is required for --retriever dense")

        return DenseRetriever.load(
            args.dense_index_dir,
            device=args.device,
        )

    if args.retriever == "hybrid":
        if not args.bm25_index_dir or not args.dense_index_dir:
            raise ValueError(
                "--bm25-index-dir and --dense-index-dir are required for --retriever hybrid"
            )

        return HybridRetriever.from_indexes(
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

    raise ValueError(f"Unsupported retriever: {args.retriever}")


def load_llm(args):
    if args.mock_llm:
        return MockLLM()

    config = LocalLLMConfig(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

    return LocalLLM(config)


def print_response(response: dict[str, Any]) -> None:
    generated = response["generated_answer"]
    verification = response["verification"]
    retrieved = response["retrieved_chunks"]

    print("\nAnswer")
    print("=" * 40)
    print(generated["answer"])

    print("\nCitations")
    print("=" * 40)
    if generated["citations"]:
        for citation in generated["citations"]:
            print(f"- {citation}")
    else:
        print("None")

    print("\nAbstention")
    print("=" * 40)
    print(f"Abstained: {generated['abstained']}")
    print(f"Reason: {generated['abstention_reason']}")

    print("\nVerification")
    print("=" * 40)
    print(f"Status: {verification['verification_status']}")

    if verification["invalid_citations"]:
        print(f"Invalid citations: {verification['invalid_citations']}")

    if verification["missing_citations"]:
        print(f"Missing citations: {verification['missing_citations']}")

    if verification["unsupported_claims"]:
        print(f"Unsupported claims: {verification['unsupported_claims']}")

    if verification["safety_warnings"]:
        print(f"Safety warnings: {verification['safety_warnings']}")

    if verification["notes"]:
        print(f"Notes: {verification['notes']}")

    print("\nRetrieved chunks")
    print("=" * 40)
    for chunk in retrieved:
        print(
            f"- rank={chunk.get('rank')} "
            f"chunk_id={chunk['chunk_id']} "
            f"section={chunk['section']} "
            f"method={chunk.get('retrieval_method')} "
            f"score={chunk.get('score')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a citation-grounded RAG answer using local/offline components."
    )

    parser.add_argument("--query", required=True, help="Question to answer")

    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
    )

    parser.add_argument("--bm25-index-dir", default="data/indexes/bm25")
    parser.add_argument("--dense-index-dir", default="data/indexes/faiss")

    parser.add_argument(
        "--fusion-method",
        choices=["rrf", "weighted"],
        default="weighted",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-candidate-k", type=int, default=30)
    parser.add_argument("--dense-candidate-k", type=int, default=30)
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument("--dense-weight", type=float, default=0.3)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-context-chunks", type=int, default=5)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1600)

    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock LLM instead of loading a real model.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Local Hugging Face model name or path",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--generation-mode",
        choices=["llm", "extractive", "extract_then_rewrite"],
        default="llm",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON response",
    )

    args = parser.parse_args()

    retriever = load_retriever(args)
    llm = None if args.generation_mode == "extractive" else load_llm(args)

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        generation_mode=args.generation_mode,
        max_context_chunks=args.max_context_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )

    response = pipeline.answer(
        question=args.query,
        top_k=args.top_k,
    )

    response_dict = response.to_dict()
    print_response(response_dict)

    if args.output:
        write_json(response_dict, args.output)
        print(f"\nSaved response to {args.output}")


if __name__ == "__main__":
    main()