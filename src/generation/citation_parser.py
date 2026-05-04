from __future__ import annotations

import re


# Matches citations like:
# [doc_001_monitoring_chunk_001]
# [doc_013_evidence_summary_chunk_001]
CITATION_RE = re.compile(r"\[(doc_\d{3}_[a-z_]+_chunk_\d{3})\]")


def extract_citations(text: str) -> list[str]:
    """
    Extract unique chunk-ID citations from generated text.

    Returns sorted unique citations for stable evaluation.
    """
    if not text:
        return []

    return sorted(set(CITATION_RE.findall(text)))


def parse_abstention(text: str) -> tuple[bool, str | None]:
    """
    Detect whether the model abstained.

    Returns:
        (abstained, abstention_reason)
    """
    if not text:
        return False, None

    lowered = text.lower()

    insufficient_patterns = [
        "provided evidence is insufficient",
        "evidence is insufficient",
        "insufficient to answer",
        "not enough information",
        "does not provide enough information",
        "cannot be determined from the provided evidence",
        "not available in the provided evidence",
    ]

    conflict_patterns = [
        "evidence is conflicting",
        "conflicting evidence",
        "evidence is inconsistent",
        "inconsistent evidence",
    ]

    ambiguity_patterns = [
        "evidence is ambiguous",
        "ambiguous evidence",
        "not consistently defined",
        "unclear from the evidence",
    ]

    if any(pattern in lowered for pattern in insufficient_patterns):
        return True, "insufficient_evidence"

    if any(pattern in lowered for pattern in conflict_patterns):
        return True, "conflicting_evidence"

    if any(pattern in lowered for pattern in ambiguity_patterns):
        return True, "ambiguous_evidence"

    return False, None


def parse_answer_text(raw_output: str) -> str:
    """
    Extract the answer section from model output if it follows the requested format.

    If no explicit Answer: section is found, return the full raw output.
    """
    if not raw_output:
        return ""

    match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Citations:|\n\s*Abstained:|\Z)",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return raw_output.strip()


def parse_declared_citations(raw_output: str) -> list[str]:
    """
    Extract citations from a model's explicit 'Citations:' line, if present.

    This is complementary to extract_citations(), which scans the whole text.
    """
    if not raw_output:
        return []

    match = re.search(
        r"Citations:\s*(.*?)(?:\n\s*Abstained:|\n\s*Abstention reason:|\Z)",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return []

    citation_block = match.group(1).strip()

    if citation_block.lower() in {"null", "none", "n/a", ""}:
        return []

    return sorted(set(CITATION_RE.findall(citation_block)))


def parse_declared_abstained(raw_output: str) -> bool | None:
    """
    Parse explicit Abstained: true|false field.

    Returns None if the field is absent.
    """
    if not raw_output:
        return None

    match = re.search(
        r"Abstained:\s*(true|false)",
        raw_output,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).lower() == "true"


def parse_declared_abstention_reason(raw_output: str) -> str | None:
    """
    Parse explicit Abstention reason field.
    """
    if not raw_output:
        return None

    match = re.search(
        r"Abstention reason:\s*(.*?)(?:\n|$)",
        raw_output,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    reason = match.group(1).strip()

    if reason.lower() in {"null", "none", "n/a", ""}:
        return None

    return reason