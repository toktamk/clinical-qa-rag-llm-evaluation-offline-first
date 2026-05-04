from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "extractive": "Extractive",
    "llm": "Direct LLM",
    "two_step": "Extract -> Rewrite",
}


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: list[str], default: float | None = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "TBD"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def fmt_ms(value: Any) -> str:
    if value is None:
        return "TBD"
    if value >= 1000:
        return f"{value:,.0f} ms"
    return f"{value:.0f} ms"


def row(label: str, report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {})
    quality = summary.get("quality") or {}

    return [
        label,
        fmt(summary.get("verification_pass_rate")),
        fmt(summary.get("abstention_accuracy")),
        fmt(summary.get("citation_precision_macro")),
        fmt(summary.get("citation_recall_macro")),
        fmt(quality.get("clarity_mean")),
        fmt(quality.get("completeness_mean")),
        fmt(quality.get("fluency_mean")),
        fmt(quality.get("overall_quality_mean")),
        fmt_ms(summary.get("latency_mean_ms")),
        fmt_ms(summary.get("latency_p95_ms")),
    ]


def make_markdown_table(rows: list[list[str]]) -> str:
    header = [
        "Method",
        "Verif ↑",
        "Abst Acc ↑",
        "Cit Prec ↑",
        "Cit Rec ↑",
        "Clarity ↑",
        "Completeness ↑",
        "Fluency ↑",
        "Quality ↑",
        "Mean Latency ↓",
        "p95 Latency ↓",
    ]
    align = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines.extend("| " + " | ".join(item) + " |" for item in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generation evaluation result JSON files and print a README-ready markdown table."
    )
    parser.add_argument("--extractive", required=True, help="Path to extractive generation result JSON")
    parser.add_argument("--llm", required=True, help="Path to direct LLM generation result JSON")
    parser.add_argument("--two-step", required=True, help="Path to extract-then-rewrite result JSON")
    parser.add_argument("--output", default=None, help="Optional path to save the markdown table")
    args = parser.parse_args()

    reports = {
        "extractive": load_json(args.extractive),
        "llm": load_json(args.llm),
        "two_step": load_json(args.two_step),
    }

    rows = [row(METHOD_LABELS[key], reports[key]) for key in ["extractive", "llm", "two_step"]]
    table = make_markdown_table(rows)

    print(table)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
        print(f"\nSaved markdown table to {output_path}")


if __name__ == "__main__":
    main()
