import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: List[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_chunks_by_doc(chunks: List[dict]) -> Dict[str, Dict[str, dict]]:
    grouped: Dict[str, Dict[str, dict]] = {}

    for chunk in chunks:
        doc_id = chunk["doc_id"]
        section = chunk["section"]

        grouped.setdefault(doc_id, {})
        grouped[doc_id][section] = chunk

    return grouped


def make_atomic_claim(question_id: str, claim_index: int, claim: str, chunk_id: str) -> dict:
    return {
        "claim_id": f"{question_id}_c{claim_index}",
        "claim": claim,
        "supported_by": [chunk_id],
        "support_type": "entailed",
    }


def make_base_item(
    question_id: str,
    question: str,
    question_type: str,
    clinical_intent: str,
    category: str,
    difficulty: str,
    gold_answer_short: str,
    gold_answer_long: str,
    doc_id: str,
    relevant_chunk_ids: List[str],
    required_citations: List[str],
    atomic_claims: List[dict],
    requires_abstention: bool,
    clinical_risk: str,
    abstention_reason: str | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "clinical_intent": clinical_intent,
        "category": category,
        "difficulty": difficulty,
        "gold_answer_short": gold_answer_short,
        "gold_answer_long": gold_answer_long,
        "source_doc_ids": [doc_id],
        "relevant_doc_ids": [doc_id],
        "relevant_chunk_ids": relevant_chunk_ids,
        "required_citations": required_citations,
        "atomic_claims": atomic_claims,
        "requires_abstention": requires_abstention,
        "abstention_reason": abstention_reason,
        "clinical_risk": clinical_risk,
        "split": None,
        "review_status": "draft",
    }


def generate_direct_monitoring_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    chunk = chunks.get("Monitoring")
    if not chunk:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"According to the guideline, what monitoring is recommended in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="monitoring",
        category="direct_evidence",
        difficulty="easy",
        gold_answer_short=chunk["text"],
        gold_answer_long=f"The guideline states the following monitoring approach: {chunk['text']}",
        doc_id=doc_id,
        relevant_chunk_ids=[chunk["chunk_id"]],
        required_citations=[chunk["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The monitoring recommendation is stated in the monitoring section.",
                chunk["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )


def generate_outcome_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    chunk = chunks.get("Outcomes")
    if not chunk:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"How does the guideline define treatment success or response in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="treatment_general_information",
        category="direct_evidence",
        difficulty="easy",
        gold_answer_short=chunk["text"],
        gold_answer_long=f"The outcomes section defines response as follows: {chunk['text']}",
        doc_id=doc_id,
        relevant_chunk_ids=[chunk["chunk_id"]],
        required_citations=[chunk["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The treatment response definition is stated in the outcomes section.",
                chunk["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )


def generate_risk_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    chunk = chunks.get("Risks")
    if not chunk:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"What risks or adverse effects are reported in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="adverse_event",
        category="clinical_risk_sensitive",
        difficulty="medium",
        gold_answer_short=chunk["text"],
        gold_answer_long=f"The risks section reports the following adverse effects or uncertainties: {chunk['text']}",
        doc_id=doc_id,
        relevant_chunk_ids=[chunk["chunk_id"]],
        required_citations=[chunk["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The adverse effects or risks are stated in the risks section.",
                chunk["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="high",
    )


def generate_population_monitoring_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    population = chunks.get("Population")
    monitoring = chunks.get("Monitoring")

    if not population or not monitoring:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"For the population described in {doc_id}, what monitoring approach is recommended?",
        question_type="multi_document_synthesis",
        clinical_intent="monitoring",
        category="multi_hop_evidence",
        difficulty="medium",
        gold_answer_short=monitoring["text"],
        gold_answer_long=(
            f"The population section defines the applicable patient group: {population['text']} "
            f"The monitoring section then recommends: {monitoring['text']}"
        ),
        doc_id=doc_id,
        relevant_chunk_ids=[population["chunk_id"], monitoring["chunk_id"]],
        required_citations=[population["chunk_id"], monitoring["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The applicable population is described in the population section.",
                population["chunk_id"],
            ),
            make_atomic_claim(
                qid,
                2,
                "The monitoring approach is described in the monitoring section.",
                monitoring["chunk_id"],
            ),
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )


def generate_contradiction_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    evidence = chunks.get("Evidence Summary")
    outcomes = chunks.get("Outcomes")

    if not evidence:
        return None

    text = evidence["text"].lower()
    if not any(term in text for term in ["inconsistent", "mixed", "inconclusive", "conflicting"]):
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    relevant_chunks = [evidence["chunk_id"]]
    if outcomes:
        relevant_chunks.insert(0, outcomes["chunk_id"])

    return make_base_item(
        question_id=qid,
        question=f"Does the evidence in {doc_id} support a single clear conclusion?",
        question_type="contradiction_detection",
        clinical_intent="treatment_general_information",
        category="conflicting_evidence",
        difficulty="hard",
        gold_answer_short="No. The document reports inconsistent or conflicting evidence.",
        gold_answer_long=(
            "The document does not support a single clear conclusion. "
            f"The evidence summary states: {evidence['text']}"
        ),
        doc_id=doc_id,
        relevant_chunk_ids=relevant_chunks,
        required_citations=relevant_chunks,
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The evidence is inconsistent or conflicting.",
                evidence["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="high",
    )

def generate_evidence_summary_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    evidence = chunks.get("Evidence Summary")
    if not evidence:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"What evidence is summarised for the recommendation in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="methodology",
        category="direct_evidence",
        difficulty="medium",
        gold_answer_short=evidence["text"],
        gold_answer_long=(
            f"The evidence summary section reports the following supporting evidence: "
            f"{evidence['text']}"
        ),
        doc_id=doc_id,
        relevant_chunk_ids=[evidence["chunk_id"]],
        required_citations=[evidence["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The supporting evidence is stated in the evidence summary section.",
                evidence["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )

def generate_temporal_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    temporal_sections = []

    for section_name in ["Intervention", "Monitoring", "Outcomes", "Evidence Summary"]:
        chunk = chunks.get(section_name)
        if not chunk:
            continue

        text_lower = chunk["text"].lower()
        has_temporal_signal = any(
            term in text_lower
            for term in [
                "hour",
                "hours",
                "day",
                "days",
                "week",
                "weeks",
                "month",
                "months",
                "year",
                "years",
                "within",
                "after",
                "before",
                "every",
            ]
        )

        if has_temporal_signal:
            temporal_sections.append(chunk)

    if not temporal_sections:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    relevant_chunk_ids = [chunk["chunk_id"] for chunk in temporal_sections]

    combined_text = " ".join(chunk["text"] for chunk in temporal_sections)

    return make_base_item(
        question_id=qid,
        question=f"What timing or follow-up schedule is described in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="monitoring",
        category="temporal_reasoning",
        difficulty="medium",
        gold_answer_short=combined_text,
        gold_answer_long=(
            f"The document describes timing or follow-up information across the following evidence: "
            f"{combined_text}"
        ),
        doc_id=doc_id,
        relevant_chunk_ids=relevant_chunk_ids,
        required_citations=relevant_chunk_ids,
        atomic_claims=[
            {
                "claim_id": f"{qid}_c1",
                "claim": "The document contains timing or follow-up schedule information.",
                "supported_by": relevant_chunk_ids,
                "support_type": "entailed",
            }
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )

def generate_citation_stress_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    """
    Generate citation-stress questions for documents containing noise, distractors,
    repeated statements, or similar-but-not-applicable recommendations.

    This is intentionally selective: it should only fire for documents whose
    metadata marks contains_noise=True.
    """
    monitoring = chunks.get("Monitoring")
    background = chunks.get("Background")
    evidence = chunks.get("Evidence Summary")

    candidate_chunks = [chunk for chunk in [monitoring, background, evidence] if chunk]

    if not candidate_chunks:
        return None

    contains_noise = any(
        chunk.get("metadata", {}).get("contains_noise", False)
        for chunk in candidate_chunks
    )

    if not contains_noise:
        return None

    if not monitoring:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    relevant_chunk_ids = [monitoring["chunk_id"]]

    # Include noisy chunks as relevant context only if they contain distractor
    # material that the model must avoid using as the answer.
    distractor_chunk_ids = [
        chunk["chunk_id"]
        for chunk in [background, evidence]
        if chunk and chunk["chunk_id"] != monitoring["chunk_id"]
    ]

    return make_base_item(
        question_id=qid,
        question=(
            f"In {doc_id}, which monitoring recommendation applies to the target condition, "
            "and what distracting information should not be treated as the recommendation?"
        ),
        question_type="citation_validation",
        clinical_intent="monitoring",
        category="citation_stress_test",
        difficulty="hard",
        gold_answer_short=monitoring["text"],
        gold_answer_long=(
            "The applicable recommendation must be taken from the monitoring section only. "
            f"The monitoring section states: {monitoring['text']} "
            "Any epidemiological, unrelated-condition, or non-applicable information should not be used "
            "as the final recommendation."
        ),
        doc_id=doc_id,
        relevant_chunk_ids=relevant_chunk_ids + distractor_chunk_ids,
        required_citations=relevant_chunk_ids,
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The applicable monitoring recommendation is stated in the monitoring section.",
                monitoring["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="high",
    )


def generate_ambiguity_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    monitoring = chunks.get("Monitoring")
    outcomes = chunks.get("Outcomes")
    evidence = chunks.get("Evidence Summary")

    candidate_text = " ".join(
        chunk["text"].lower()
        for chunk in [monitoring, outcomes, evidence]
        if chunk
    )

    if not any(term in candidate_text for term in ["not standardised", "inconsistent", "clinically meaningful", "not consistently defined"]):
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    relevant = [c["chunk_id"] for c in [monitoring, outcomes, evidence] if c]

    return make_base_item(
        question_id=qid,
        question=f"What uncertainty or ambiguity is described in {doc_id}?",
        question_type="evidence_grounded_qa",
        clinical_intent="methodology",
        category="ambiguous_evidence",
        difficulty="hard",
        gold_answer_short="The document reports inconsistent thresholds, definitions, or monitoring intervals.",
        gold_answer_long=(
            "The document describes ambiguity because thresholds, definitions, or monitoring intervals "
            "are not consistently standardised across the evidence."
        ),
        doc_id=doc_id,
        relevant_chunk_ids=relevant,
        required_citations=relevant,
        atomic_claims=[
            {
                "claim_id": f"{qid}_c1",
                "claim": "The document contains ambiguous or inconsistent definitions.",
                "supported_by": relevant,
                "support_type": "partial",
            }
        ],
        requires_abstention=False,
        clinical_risk="medium",
    )


def generate_abstention_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    full_text = " ".join(chunk["text"].lower() for chunk in chunks.values())

    abstention_terms = [
        "no standard monitoring recommendations",
        "no reliable outcome data",
        "potential risks are unknown",
        "not formally evaluated",
        "limited evidence",
    ]

    if not any(term in full_text for term in abstention_terms):
        return None

    monitoring = chunks.get("Monitoring")
    outcomes = chunks.get("Outcomes")
    evidence = chunks.get("Evidence Summary")

    relevant = [c["chunk_id"] for c in [monitoring, outcomes, evidence] if c]

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"What standard monitoring or outcome recommendation is available for the intervention in {doc_id}?",
        question_type="unanswerable",
        clinical_intent="monitoring",
        category="insufficient_evidence",
        difficulty="medium",
        gold_answer_short="The available evidence is insufficient to provide a standard recommendation.",
        gold_answer_long=(
            "The document states that standard recommendations or reliable outcome data are not available. "
            "Therefore, the correct system behaviour is to abstain rather than infer a recommendation."
        ),
        doc_id=doc_id,
        relevant_chunk_ids=relevant,
        required_citations=relevant,
        atomic_claims=[
            {
                "claim_id": f"{qid}_c1",
                "claim": "The available evidence is insufficient to provide a standard recommendation.",
                "supported_by": relevant,
                "support_type": "entailed",
            }
        ],
        requires_abstention=True,
        abstention_reason="insufficient_evidence",
        clinical_risk="high",
    )

def generate_intervention_qa(doc_id: str, chunks: Dict[str, dict], idx: int) -> dict | None:
    intervention = chunks.get("Intervention")
    if not intervention:
        return None

    qid = f"q_{doc_id}_{idx:03d}"

    return make_base_item(
        question_id=qid,
        question=f"According to the guideline, how is the intervention described in {doc_id} administered or adjusted?",
        question_type="evidence_grounded_qa",
        clinical_intent="treatment_general_information",
        category="direct_evidence",
        difficulty="easy",
        gold_answer_short=intervention["text"],
        gold_answer_long=(
            f"The intervention section describes administration or adjustment as follows: "
            f"{intervention['text']}"
        ),
        doc_id=doc_id,
        relevant_chunk_ids=[intervention["chunk_id"]],
        required_citations=[intervention["chunk_id"]],
        atomic_claims=[
            make_atomic_claim(
                qid,
                1,
                "The intervention administration or adjustment is stated in the intervention section.",
                intervention["chunk_id"],
            )
        ],
        requires_abstention=False,
        clinical_risk="high",
    )

def generate_qa_items(chunks: List[dict], items_per_doc: int = 6) -> List[dict]:
    grouped = group_chunks_by_doc(chunks)
    qa_items: List[dict] = []

    generators = [
        generate_direct_monitoring_qa,
        generate_outcome_qa,
        generate_risk_qa,
        generate_population_monitoring_qa,
        generate_intervention_qa,
        generate_evidence_summary_qa,
        generate_temporal_qa,
        generate_citation_stress_qa,
        generate_contradiction_qa,
        generate_ambiguity_qa,
        generate_abstention_qa,
    ]

    for doc_id, doc_chunks in sorted(grouped.items()):
        doc_count = 0
        idx = 1

        for generator in generators:
            if doc_count >= items_per_doc:
                break

            item = generator(doc_id, doc_chunks, idx)
            if item is not None:
                qa_items.append(item)
                doc_count += 1
                idx += 1

    return qa_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate draft clinical RAG QA items from chunks.jsonl."
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output", required=True, help="Path to output draft QA JSONL")
    parser.add_argument("--items-per-doc", type=int, default=6)

    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    qa_items = generate_qa_items(chunks, items_per_doc=args.items_per_doc)
    write_jsonl(qa_items, args.output)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Generated {len(qa_items)} draft QA items")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()