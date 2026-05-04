from __future__ import annotations

import re

from src.generation.schemas import GeneratedAnswer, VerificationResult


RECOMMENDATION_TERMS = {
    "should",
    "must",
    "recommended",
    "requires",
    "require",
    "administer",
    "initiate",
    "increase",
    "decrease",
    "monitor",
    "discontinue",
    "reinitiate",
}


PERSONALISED_MEDICAL_TERMS = {
    "you should",
    "you must",
    "your doctor should",
    "i recommend",
    "take this",
    "start taking",
    "stop taking",
}


VALID_CITATION_RE = re.compile(r"doc_\d{3}_[a-z_]+_chunk_\d{3}")


def verify_answer(
    answer: GeneratedAnswer,
    *,
    used_chunk_ids: list[str] | None = None,
) -> VerificationResult:
    """
    Deterministically verify citation and safety properties of a generated answer.

    This verifier checks:
    - cited chunks were available in the supplied context
    - non-abstained answers contain citations
    - abstained answers are allowed to have no citations
    - citation-like strings follow the expected chunk ID format
    - obvious personalised medical advice is not present

    It does not yet perform semantic entailment.
    """
    allowed_chunk_ids = set(used_chunk_ids or answer.used_chunk_ids)
    cited_chunk_ids = set(answer.citations)

    invalid_citations: list[str] = []
    missing_citations: list[str] = []
    unsupported_claims: list[str] = []
    safety_warnings: list[str] = []
    notes: list[str] = []

    answer_text = answer.answer or ""

    # 1. Citation validity against provided context.
    out_of_context_citations = sorted(cited_chunk_ids - allowed_chunk_ids)
    invalid_citations.extend(out_of_context_citations)

    # 2. Non-abstained answers must cite evidence.
    if not answer.abstained and not answer.citations:
        missing_citations.append("Non-abstained answer has no citations.")

    # 3. Abstention consistency.
    if answer.abstained:
        if not answer.abstention_reason:
            notes.append("Abstained answer has no abstention_reason.")

        if answer.citations:
            notes.append(
                "Abstained answer includes citations; this is acceptable if citing evidence insufficiency."
            )

    # 4. Recommendation-style language is only an error for non-abstained answers.
    if not answer.abstained:
        if contains_recommendation_language(answer_text) and not answer.citations:
            unsupported_claims.append(
                "Answer appears to contain recommendation-style language without citations."
            )

    # 5. Personalised medical advice is always unsafe, even in abstention text.
    if contains_personalised_medical_advice(answer_text):
        safety_warnings.append(
            "Answer may contain personalised medical advice, which is outside system scope."
        )

    # 6. Detect malformed citation-like strings.
    malformed_citations = find_malformed_citation_like_strings(answer_text)
    invalid_citations.extend(malformed_citations)

    # 7. Determine status.
    if invalid_citations or missing_citations or unsupported_claims or safety_warnings:
        status = "failed"
    elif notes:
        status = "warning"
    else:
        status = "passed"

    return VerificationResult(
        verification_status=status,
        unsupported_claims=unsupported_claims,
        missing_citations=missing_citations,
        invalid_citations=sorted(set(invalid_citations)),
        safety_warnings=safety_warnings,
        notes=notes,
    )


def contains_recommendation_language(text: str) -> bool:
    """
    Detect generic recommendation-style language.

    This is a lightweight heuristic, not a clinical safety classifier.
    """
    lowered = text.lower()
    tokens = set(re.findall(r"\b[a-z]+\b", lowered))
    return bool(tokens & RECOMMENDATION_TERMS)


def contains_personalised_medical_advice(text: str) -> bool:
    """
    Detect personalised medical advice phrasing.

    The system should discuss provided evidence, not advise an individual user.
    """
    lowered = text.lower()
    return any(term in lowered for term in PERSONALISED_MEDICAL_TERMS)


def find_malformed_citation_like_strings(text: str) -> list[str]:
    """
    Detect bracketed citation-like content that starts with doc_ but does not
    match the valid chunk ID format.

    Valid:
        [doc_001_monitoring_chunk_001]
    """
    if not text:
        return []

    malformed: list[str] = []
    bracketed_items = re.findall(r"\[([^\]]+)\]", text)

    for item in bracketed_items:
        item = item.strip()

        if item.startswith("doc_") and not VALID_CITATION_RE.fullmatch(item):
            malformed.append(item)

    return malformed