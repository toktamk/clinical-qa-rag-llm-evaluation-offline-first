import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_CATEGORIES = {
    "direct_evidence",
    "multi_hop_evidence",
    "citation_stress_test",
    "conflicting_evidence",
    "ambiguous_evidence",
    "insufficient_evidence",
    "clinical_risk_sensitive",
    "temporal_reasoning",
    "out_of_domain",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_REVIEW_STATUS = {"draft", "needs_review", "validated", "rejected"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_SUPPORT_TYPES = {"entailed", "partial", "contradicted", "insufficient"}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(data: dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def validate_required_fields(item: dict[str, Any]) -> list[str]:
    required = [
        "schema_version",
        "question_id",
        "question",
        "question_type",
        "clinical_intent",
        "category",
        "difficulty",
        "gold_answer_short",
        "gold_answer_long",
        "source_doc_ids",
        "relevant_doc_ids",
        "relevant_chunk_ids",
        "required_citations",
        "atomic_claims",
        "requires_abstention",
        "clinical_risk",
        "review_status",
    ]

    errors = []

    for field in required:
        if field not in item:
            errors.append(f"Missing required field: {field}")

    return errors


def validate_item(
    item: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
    question_id_counts: Counter,
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    errors.extend(validate_required_fields(item))

    question_id = item.get("question_id", "<missing_question_id>")

    if question_id_counts[question_id] > 1:
        errors.append(f"Duplicate question_id: {question_id}")

    question = item.get("question", "")
    if not isinstance(question, str) or not question.strip():
        errors.append("Question is empty.")
    elif not question.strip().endswith("?"):
        warnings.append("Question does not end with '?'.")

    if item.get("category") not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {item.get('category')}")

    if item.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append(f"Invalid difficulty: {item.get('difficulty')}")

    if item.get("review_status") not in VALID_REVIEW_STATUS:
        errors.append(f"Invalid review_status: {item.get('review_status')}")

    if item.get("clinical_risk") not in VALID_RISK_LEVELS:
        errors.append(f"Invalid clinical_risk: {item.get('clinical_risk')}")

    relevant_chunk_ids = item.get("relevant_chunk_ids", [])
    required_citations = item.get("required_citations", [])

    if not isinstance(relevant_chunk_ids, list):
        errors.append("relevant_chunk_ids must be a list.")
        relevant_chunk_ids = []

    if not isinstance(required_citations, list):
        errors.append("required_citations must be a list.")
        required_citations = []

    for chunk_id in relevant_chunk_ids:
        if chunk_id not in chunk_index:
            errors.append(f"Unknown relevant_chunk_id: {chunk_id}")

    for chunk_id in required_citations:
        if chunk_id not in chunk_index:
            errors.append(f"Unknown required citation chunk_id: {chunk_id}")

    if not item.get("requires_abstention", False):
        if len(required_citations) == 0:
            errors.append("Answerable item has no required_citations.")

    if item.get("requires_abstention", False):
        if item.get("category") != "insufficient_evidence":
            warnings.append(
                "requires_abstention=True but category is not insufficient_evidence."
            )
        if not item.get("abstention_reason"):
            warnings.append("Abstention item has no abstention_reason.")

    atomic_claims = item.get("atomic_claims", [])

    if not isinstance(atomic_claims, list):
        errors.append("atomic_claims must be a list.")
        atomic_claims = []

    if len(atomic_claims) == 0:
        errors.append("Item has no atomic_claims.")

    for claim in atomic_claims:
        claim_id = claim.get("claim_id")
        claim_text = claim.get("claim")
        supported_by = claim.get("supported_by", [])
        support_type = claim.get("support_type")

        if not claim_id:
            errors.append("Atomic claim missing claim_id.")

        if not claim_text:
            errors.append("Atomic claim missing claim text.")

        if support_type not in VALID_SUPPORT_TYPES:
            errors.append(f"Invalid support_type: {support_type}")

        if not isinstance(supported_by, list):
            errors.append("Atomic claim supported_by must be a list.")
            supported_by = []

        if len(supported_by) == 0:
            errors.append("Atomic claim has empty supported_by list.")

        for chunk_id in supported_by:
            if chunk_id not in chunk_index:
                errors.append(f"Atomic claim references unknown chunk_id: {chunk_id}")

    # Consistency check: citations should usually be subset of relevant chunks.
    citation_not_relevant = set(required_citations) - set(relevant_chunk_ids)
    if citation_not_relevant:
        warnings.append(
            f"required_citations not included in relevant_chunk_ids: "
            f"{sorted(citation_not_relevant)}"
        )

    # Useful quality warning.
    if len(item.get("gold_answer_short", "")) > len(item.get("gold_answer_long", "")):
        warnings.append("gold_answer_short is longer than gold_answer_long.")

    return errors, warnings


def summarise_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_items": len(items),
        "category_counts": dict(Counter(item.get("category") for item in items)),
        "difficulty_counts": dict(Counter(item.get("difficulty") for item in items)),
        "review_status_counts": dict(Counter(item.get("review_status") for item in items)),
        "abstention_count": sum(1 for item in items if item.get("requires_abstention")),
        "clinical_risk_counts": dict(Counter(item.get("clinical_risk") for item in items)),
    }


def validate_dataset(eval_file: str, chunks_file: str) -> dict[str, Any]:
    items = load_jsonl(eval_file)
    chunks = load_jsonl(chunks_file)
    chunk_index = build_chunk_index(chunks)

    question_id_counts = Counter(item.get("question_id") for item in items)

    item_reports = []
    total_errors = 0
    total_warnings = 0

    for item in items:
        question_id = item.get("question_id", "<missing_question_id>")
        errors, warnings = validate_item(item, chunk_index, question_id_counts)

        if errors or warnings:
            item_reports.append(
                {
                    "question_id": question_id,
                    "errors": errors,
                    "warnings": warnings,
                }
            )

        total_errors += len(errors)
        total_warnings += len(warnings)

    report = {
        "eval_file": eval_file,
        "chunks_file": chunks_file,
        "summary": summarise_items(items),
        "chunk_count": len(chunks),
        "unique_chunk_count": len(chunk_index),
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "passed": total_errors == 0,
        "items_with_issues": item_reports,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    print("\nEvaluation dataset validation")
    print("=" * 40)
    print(f"Items: {report['summary']['total_items']}")
    print(f"Chunks: {report['chunk_count']}")
    print(f"Errors: {report['total_errors']}")
    print(f"Warnings: {report['total_warnings']}")
    print(f"Passed: {report['passed']}")

    print("\nCategory counts:")
    for category, count in sorted(report["summary"]["category_counts"].items()):
        print(f"  {category}: {count}")

    print("\nDifficulty counts:")
    for difficulty, count in sorted(report["summary"]["difficulty_counts"].items()):
        print(f"  {difficulty}: {count}")

    print("\nReview status counts:")
    for status, count in sorted(report["summary"]["review_status_counts"].items()):
        print(f"  {status}: {count}")

    if report["items_with_issues"]:
        print("\nItems with issues:")
        for item in report["items_with_issues"][:20]:
            print(f"\n- {item['question_id']}")
            for error in item["errors"]:
                print(f"  ERROR: {error}")
            for warning in item["warnings"]:
                print(f"  WARNING: {warning}")

        remaining = len(report["items_with_issues"]) - 20
        if remaining > 0:
            print(f"\n... {remaining} more items with issues. See JSON report.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate clinical RAG evaluation dataset against chunks.jsonl."
    )
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    report = validate_dataset(args.eval_file, args.chunks)
    write_json(report, args.output)
    print_report(report)

    if report["passed"]:
        print(f"\nValidation passed. Report saved to {args.output}")
    else:
        print(f"\nValidation failed. Report saved to {args.output}")


if __name__ == "__main__":
    main()