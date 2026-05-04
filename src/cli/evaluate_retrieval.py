from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.evaluation.retrieval_metrics import (
    aggregate_metrics,
    evaluate_single_query,
    get_retrieved_chunk_ids,
)
from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_retriever(
    method: str,
    bm25_index_dir: str | None,
    dense_index_dir: str | None,
    device: str,
    fusion_method: str,
    rrf_k: int,
    bm25_candidate_k: int,
    dense_candidate_k: int,
    bm25_weight: float,
    dense_weight: float,
):
    if method == "bm25":
        if not bm25_index_dir:
            raise ValueError("--bm25-index-dir is required for method=bm25")
        return BM25Retriever.load(bm25_index_dir)

    if method == "dense":
        if not dense_index_dir:
            raise ValueError("--dense-index-dir is required for method=dense")
        return DenseRetriever.load(dense_index_dir, device=device)

    if method == "hybrid":
        if not bm25_index_dir or not dense_index_dir:
            raise ValueError(
                "--bm25-index-dir and --dense-index-dir are required for method=hybrid"
            )

        return HybridRetriever.from_indexes(
            bm25_index_dir=bm25_index_dir,
            dense_index_dir=dense_index_dir,
            device=device,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            bm25_candidate_k=bm25_candidate_k,
            dense_candidate_k=dense_candidate_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    raise ValueError(f"Unsupported retrieval method: {method}")


def get_relevant_ids(item: dict[str, Any], relevance_field: str) -> list[str]:
    values = item.get(relevance_field, [])

    if not isinstance(values, list):
        raise ValueError(
            f"{relevance_field} must be a list for question_id={item.get('question_id')}"
        )

    return values


def should_skip_item(item: dict[str, Any], relevance_field: str) -> bool:
    relevant_ids = item.get(relevance_field, [])

    if not relevant_ids:
        return True

    # Out-of-domain items often intentionally have no gold chunks.
    if item.get("category") == "out_of_domain":
        return True

    return False


def evaluate_retriever(
    retriever,
    eval_items: list[dict[str, Any]],
    *,
    method: str,
    k_values: list[int],
    top_k: int,
    relevance_field: str,
) -> dict[str, Any]:
    per_query_metrics = []
    per_query_outputs = []
    latencies_ms = []

    evaluated_count = 0
    skipped_count = 0

    for item in eval_items:
        question_id = item.get("question_id", "<missing_question_id>")
        question = item.get("question", "")

        if should_skip_item(item, relevance_field):
            skipped_count += 1
            continue

        relevant_ids = get_relevant_ids(item, relevance_field)

        start = time.perf_counter()
        results = retriever.retrieve(question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_ids = get_retrieved_chunk_ids(results)

        metrics = evaluate_single_query(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k_values=k_values,
        )

        per_query_metrics.append(metrics)
        latencies_ms.append(latency_ms)
        evaluated_count += 1

        per_query_outputs.append(
            {
                "question_id": question_id,
                "question": question,
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
                "requires_abstention": item.get("requires_abstention"),
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "latency_ms": latency_ms,
                "metrics": metrics,
                "top_results": [
                    {
                        "rank": result.rank,
                        "chunk_id": result.chunk_id,
                        "doc_id": result.doc_id,
                        "section": result.section,
                        "score": result.score,
                        "retrieval_method": result.retrieval_method,
                    }
                    for result in results
                ],
            }
        )

    aggregate = aggregate_metrics(per_query_metrics)

    if latencies_ms:
        sorted_latencies = sorted(latencies_ms)
        aggregate["latency_mean_ms"] = sum(latencies_ms) / len(latencies_ms)
        aggregate["latency_p50_ms"] = percentile(sorted_latencies, 50)
        aggregate["latency_p95_ms"] = percentile(sorted_latencies, 95)
    else:
        aggregate["latency_mean_ms"] = 0.0
        aggregate["latency_p50_ms"] = 0.0
        aggregate["latency_p95_ms"] = 0.0

    return {
        "method": method,
        "k_values": k_values,
        "top_k": top_k,
        "relevance_field": relevance_field,
        "evaluated_count": evaluated_count,
        "skipped_count": skipped_count,
        "aggregate_metrics": aggregate,
        "per_query": per_query_outputs,
    }


def percentile(sorted_values: list[float], percentile_value: int) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * (percentile_value / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def print_summary(report: dict[str, Any]) -> None:
    print("\nRetrieval evaluation")
    print("=" * 40)
    print(f"Method: {report['method']}")
    print(f"Evaluated: {report['evaluated_count']}")
    print(f"Skipped: {report['skipped_count']}")

    print("\nAggregate metrics:")
    for name, value in sorted(report["aggregate_metrics"].items()):
        print(f"  {name}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against QA dataset.")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--method", required=True, choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--output", required=True)

    parser.add_argument("--bm25-index-dir", default=None)
    parser.add_argument("--dense-index-dir", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--relevance-field",
        default="relevant_chunk_ids",
        choices=["relevant_chunk_ids", "required_citations"],
    )

    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-candidate-k", type=int, default=30)
    parser.add_argument("--dense-candidate-k", type=int, default=30)
    parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument("--dense-weight", type=float, default=0.3)

    args = parser.parse_args()

    eval_items = load_jsonl(args.eval_file)

    retriever = load_retriever(
        method=args.method,
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

    report = evaluate_retriever(
        retriever,
        eval_items,
        method=args.method,
        k_values=args.k_values,
        top_k=args.top_k,
        relevance_field=args.relevance_field,
    )

    write_json(report, args.output)
    print_summary(report)
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()