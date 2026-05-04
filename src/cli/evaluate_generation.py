from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.evaluation.llm_judge import LLMJudge
from src.generation.local_llm import LocalLLM, LocalLLMConfig, MockLLM
from src.generation.rag_pipeline import RAGPipeline
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


def load_retriever(args: argparse.Namespace):
    if args.retriever == "bm25":
        return BM25Retriever.load(args.bm25_index_dir)

    if args.retriever == "dense":
        return DenseRetriever.load(args.dense_index_dir, device=args.device)

    if args.retriever == "hybrid":
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


def load_llm(args: argparse.Namespace):
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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def count_by_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    for item in items:
        key = str(item.get(field))
        counts[key] = counts.get(key, 0) + 1

    return counts


def compute_citation_metrics(
    predicted_citations: list[str],
    required_citations: list[str],
    used_chunk_ids: list[str],
) -> dict[str, float | int]:
    predicted = set(predicted_citations)
    required = set(required_citations)
    used = set(used_chunk_ids)

    valid_predicted = predicted & used
    correct_required = predicted & required

    return {
        "predicted_citation_count": len(predicted),
        "required_citation_count": len(required),
        "valid_predicted_citation_count": len(valid_predicted),
        "correct_required_citation_count": len(correct_required),
        "citation_precision": safe_divide(len(valid_predicted), len(predicted)),
        "citation_recall": safe_divide(len(correct_required), len(required)),
    }


def evaluate_item(
    item: dict[str, Any],
    pipeline: RAGPipeline,
    *,
    top_k: int,
    judge: LLMJudge | None = None,
) -> dict[str, Any]:
    question = item["question"]
    question_id = item.get("question_id", "<missing_question_id>")

    required_citations = item.get("required_citations", [])
    gold_requires_abstention = bool(item.get("requires_abstention", False))

    start = time.perf_counter()
    response = pipeline.answer(question=question, top_k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000.0

    response_dict = response.to_dict()
    generated = response_dict["generated_answer"]
    verification = response_dict["verification"]

    predicted_citations = generated["citations"]
    used_chunk_ids = generated["used_chunk_ids"]
    predicted_abstained = bool(generated["abstained"])

    citation_metrics = compute_citation_metrics(
        predicted_citations=predicted_citations,
        required_citations=required_citations,
        used_chunk_ids=used_chunk_ids,
    )

    abstention_correct = predicted_abstained == gold_requires_abstention

    forced_abstention = bool(
        generated.get("metadata", {}).get("enforced_abstention", False)
    )

    verification_passed = verification["verification_status"] == "passed"

    judge_score = None
    if judge is not None:
        judge_score = judge.score(
            question=question,
            answer=generated["answer"],
            gold_answer=item.get("gold_answer_long"),
        ).to_dict()

    return {
        "question_id": question_id,
        "question": question,
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "gold_requires_abstention": gold_requires_abstention,
        "predicted_abstained": predicted_abstained,
        "abstention_correct": abstention_correct,
        "forced_abstention": forced_abstention,
        "verification_status": verification["verification_status"],
        "verification_passed": verification_passed,
        "latency_ms": latency_ms,
        "required_citations": required_citations,
        "predicted_citations": predicted_citations,
        "used_chunk_ids": used_chunk_ids,
        "citation_metrics": citation_metrics,
        "answer": generated["answer"],
        "raw_model_output": generated["raw_model_output"],
        "retrieved_chunk_ids": [
            chunk["chunk_id"] for chunk in response_dict["retrieved_chunks"]
        ],
        "verification": verification,
        "llm_judge": judge_score,
    }


def aggregate_quality_metrics(per_item: list[dict[str, Any]]) -> dict[str, float] | None:
    judge_items = [
        item["llm_judge"]
        for item in per_item
        if item.get("llm_judge") is not None
    ]

    if not judge_items:
        return None

    return {
        "clarity_mean": safe_mean([x["clarity"] for x in judge_items]),
        "completeness_mean": safe_mean([x["completeness"] for x in judge_items]),
        "fluency_mean": safe_mean([x["fluency"] for x in judge_items]),
        "overall_quality_mean": safe_mean([x["overall"] for x in judge_items]),
    }


def aggregate_results(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_item:
        return {}

    n = len(per_item)

    citation_precision_values = [
        item["citation_metrics"]["citation_precision"] for item in per_item
    ]
    citation_recall_values = [
        item["citation_metrics"]["citation_recall"] for item in per_item
    ]

    latencies = sorted(item["latency_ms"] for item in per_item)

    verification_pass_count = sum(1 for item in per_item if item["verification_passed"])
    abstention_correct_count = sum(1 for item in per_item if item["abstention_correct"])
    forced_abstention_count = sum(1 for item in per_item if item["forced_abstention"])
    predicted_abstention_count = sum(1 for item in per_item if item["predicted_abstained"])
    gold_abstention_count = sum(1 for item in per_item if item["gold_requires_abstention"])

    non_abstained_items = [
        item for item in per_item if not item["predicted_abstained"]
    ]

    citation_precision_non_abstained = [
        item["citation_metrics"]["citation_precision"]
        for item in non_abstained_items
    ]

    citation_recall_non_abstained = [
        item["citation_metrics"]["citation_recall"]
        for item in non_abstained_items
    ]

    return {
        "total_items": n,
        "verification_pass_rate": verification_pass_count / n,
        "abstention_accuracy": abstention_correct_count / n,
        "gold_abstention_rate": gold_abstention_count / n,
        "predicted_abstention_rate": predicted_abstention_count / n,
        "forced_abstention_rate": forced_abstention_count / n,
        "citation_precision_macro": safe_mean(citation_precision_values),
        "citation_recall_macro": safe_mean(citation_recall_values),
        "citation_precision_non_abstained_macro": safe_mean(
            citation_precision_non_abstained
        ),
        "citation_recall_non_abstained_macro": safe_mean(
            citation_recall_non_abstained
        ),
        "latency_mean_ms": safe_mean(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "quality": aggregate_quality_metrics(per_item),
        "category_counts": count_by_field(per_item, "category"),
        "difficulty_counts": count_by_field(per_item, "difficulty"),
        "verification_status_counts": count_by_field(per_item, "verification_status"),
    }


def filter_items(
    items: list[dict[str, Any]],
    *,
    max_items: int | None,
    category: str | None,
    include_abstention_only: bool,
) -> list[dict[str, Any]]:
    filtered = items

    if category:
        filtered = [item for item in filtered if item.get("category") == category]

    if include_abstention_only:
        filtered = [item for item in filtered if item.get("requires_abstention")]

    if max_items is not None:
        filtered = filtered[:max_items]

    return filtered


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]

    print("\nGeneration evaluation")
    print("=" * 40)
    print(f"Items: {summary['total_items']}")
    print(f"Verification pass rate: {summary['verification_pass_rate']:.4f}")
    print(f"Abstention accuracy: {summary['abstention_accuracy']:.4f}")
    print(f"Gold abstention rate: {summary['gold_abstention_rate']:.4f}")
    print(f"Predicted abstention rate: {summary['predicted_abstention_rate']:.4f}")
    print(f"Forced abstention rate: {summary['forced_abstention_rate']:.4f}")
    print(f"Citation precision macro: {summary['citation_precision_macro']:.4f}")
    print(f"Citation recall macro: {summary['citation_recall_macro']:.4f}")

    print(
        "Citation precision non-abstained: "
        f"{summary['citation_precision_non_abstained_macro']:.4f}"
    )
    print(
        "Citation recall non-abstained: "
        f"{summary['citation_recall_non_abstained_macro']:.4f}"
    )

    print(f"Latency mean ms: {summary['latency_mean_ms']:.2f}")
    print(f"Latency p95 ms: {summary['latency_p95_ms']:.2f}")

    if summary.get("quality") is not None:
        quality = summary["quality"]
        print("\nAnswer quality (LLM judge)")
        print("=" * 40)
        print(f"Clarity mean: {quality['clarity_mean']:.4f}")
        print(f"Completeness mean: {quality['completeness_mean']:.4f}")
        print(f"Fluency mean: {quality['fluency_mean']:.4f}")
        print(f"Overall quality mean: {quality['overall_quality_mean']:.4f}")

    print("\nVerification status counts:")
    for key, value in summary["verification_status_counts"].items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate citation-grounded RAG generation over QA dataset."
    )

    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output", required=True)

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

    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")

    parser.add_argument(
        "--generation-mode",
        choices=["llm", "extractive", "extract_then_rewrite"],
        default="llm",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Use local LLM judge to score clarity, completeness and fluency.",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for quick evaluation runs.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter, e.g. insufficient_evidence.",
    )
    parser.add_argument(
        "--abstention-only",
        action="store_true",
        help="Evaluate only items whose gold label requires abstention.",
    )

    args = parser.parse_args()

    eval_items = load_jsonl(args.eval_file)
    eval_items = filter_items(
        eval_items,
        max_items=args.max_items,
        category=args.category,
        include_abstention_only=args.abstention_only,
    )

    if not eval_items:
        raise ValueError("No evaluation items selected.")

    retriever = load_retriever(args)

    llm_required = args.generation_mode in {"llm", "extract_then_rewrite"}
    judge_required = args.enable_llm_judge

    llm = load_llm(args) if llm_required or judge_required else None
    judge = LLMJudge(llm) if args.enable_llm_judge and llm is not None else None

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        generation_mode=args.generation_mode,
        max_context_chunks=args.max_context_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )

    per_item: list[dict[str, Any]] = []

    for index, item in enumerate(eval_items, start=1):
        question_id = item.get("question_id", f"item_{index}")
        print(f"[{index}/{len(eval_items)}] Evaluating {question_id}")

        result = evaluate_item(
            item,
            pipeline,
            top_k=args.top_k,
            judge=judge,
        )

        per_item.append(result)

    report = {
        "eval_file": args.eval_file,
        "model": "mock" if args.mock_llm else args.model,
        "retriever": args.retriever,
        "generation_mode": args.generation_mode,
        "fusion_method": args.fusion_method if args.retriever == "hybrid" else None,
        "top_k": args.top_k,
        "max_context_chunks": args.max_context_chunks,
        "llm_judge_enabled": args.enable_llm_judge,
        "summary": aggregate_results(per_item),
        "per_item": per_item,
    }

    write_json(report, args.output)
    print_summary(report)
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()