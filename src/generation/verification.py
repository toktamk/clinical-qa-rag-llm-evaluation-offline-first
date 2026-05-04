def verify_answer(answer: str, chunks: list, required_citations: list) -> bool:
    """
    Simple but effective verification layer:
    - Check citations are present
    - Check answer overlaps with evidence
    """

    # Check citations exist
    for c in required_citations:
        if str(c) not in answer:
            return False

    # Basic grounding check
    evidence_text = " ".join([c["text"] for c in chunks])

    overlap = sum(1 for word in answer.split() if word in evidence_text)

    grounding_score = overlap / max(len(answer.split()), 1)

    return grounding_score > 0.3