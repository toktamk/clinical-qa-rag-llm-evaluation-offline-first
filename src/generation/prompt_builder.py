from __future__ import annotations


SYSTEM_PROMPT = """You are an evidence-grounded clinical assistant.

You MUST follow these rules strictly:

1. Every factual statement MUST include a citation in the form [CHUNK_ID].
2. If you cannot cite a statement, you MUST NOT include it.
3. If you cannot answer fully with citations, you MUST say exactly:
   "The provided evidence is insufficient to answer this question."
4. Do NOT output any factual claim without a citation.
5. The answer is INVALID if citations are missing.

Failure to follow these rules means the answer is incorrect.
"""

ANSWER_FORMAT = """Answer:
- <statement> [doc_XXX]
- <statement> [doc_XXX]

Citations:
- <doc_XXX>

Abstained: <true|false>
Abstention reason: <reason or null>
"""

def build_generation_prompt(question: str, context_text: str) -> str:
    """
    Build the full generation prompt.

    The prompt is intentionally strict because this project evaluates:
    - groundedness
    - citation correctness
    - abstention behaviour
    - hallucination resistance
    """
    return f"""{SYSTEM_PROMPT}

Question:
{question}

Evidence:
{context_text}

{ANSWER_FORMAT}
"""


def build_abstention_only_prompt(question: str, context_text: str) -> str:
    """
    Optional stricter prompt for known insufficient-evidence examples.
    Useful during evaluation and debugging.
    """
    return f"""{SYSTEM_PROMPT}

The evidence may be incomplete. If the answer cannot be fully supported, abstain.

Question:
{question}

Evidence:
{context_text}

{ANSWER_FORMAT}
"""


def build_verification_prompt(question: str, answer: str, context_text: str) -> str:
    """
    Prompt for optional LLM-based verification later.

    This is not used in the first deterministic verifier, but is included so the
    system can later support agentic verification.
    """
    return f"""You are verifying whether an answer is fully supported by evidence.

Question:
{question}

Answer:
{answer}

Evidence:
{context_text}

Check:
1. Are all factual claims supported by the evidence?
2. Are citations valid chunk IDs from the evidence?
3. Does the answer make unsupported medical recommendations?
4. Should the answer have abstained?

Return:
Verification: passed|failed
Unsupported claims: <list or null>
Invalid citations: <list or null>
Notes: <brief notes>
"""