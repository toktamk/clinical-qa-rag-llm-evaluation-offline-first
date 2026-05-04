def build_rewrite_prompt(
    query: str,
    extractive_answer: str,
    citations: list,
    chunks: list
) -> str:
    """
    Strict rewrite prompt:
    - No hallucination
    - Must preserve citations
    - Must stay grounded
    """

    context_text = "\n\n".join([
        f"[{i}] {c['text']}" for i, c in enumerate(chunks)
    ])

    citation_str = ", ".join([str(c) for c in citations])

    prompt = f"""
You are a clinical AI assistant.

Rewrite the answer using ONLY the provided evidence.

STRICT RULES:
- Do NOT add new information
- Do NOT hallucinate
- Keep all citations exactly as provided
- If unsure, say: "I do not know"
- Use formal clinical tone

QUESTION:
{query}

EXTRACTED ANSWER:
{extractive_answer}

CITATIONS:
{citation_str}

EVIDENCE:
{context_text}

FINAL ANSWER (WITH CITATIONS):
"""

    return prompt.strip()