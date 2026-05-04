from __future__ import annotations

import re
import string


_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*|[≥≤<>]=?|%|\d+(?:\.\d+)?")


def normalise_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    normalise_whitespace: bool = True,
) -> str:
    """
    Normalise text for retrieval.

    Args:
        text: Input text.
        lowercase: Convert to lowercase.
        remove_punctuation: Replace punctuation with spaces.
        normalise_whitespace: Collapse repeated whitespace.

    Returns:
        Normalised text string.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")

    # Normalise common dash variants.
    text = text.replace("–", "-").replace("—", "-")

    # Preserve clinically useful comparison symbols.
    text = text.replace("≥", " greater than or equal to ")
    text = text.replace("≤", " less than or equal to ")

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = text.translate(_PUNCT_TRANSLATION)

    if normalise_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenise text for BM25.

    This tokenizer keeps:
    - numbers
    - percentages
    - comparison terms
    - therapy identifiers
    - condition identifiers

    It is intentionally simple and dependency-free.
    """
    text = normalise_text(
        text,
        lowercase=True,
        remove_punctuation=False,
        normalise_whitespace=True,
    )

    tokens = _TOKEN_RE.findall(text)

    # Remove very low-signal tokens while keeping clinical/numeric terms.
    return [token for token in tokens if token and token not in STOPWORDS]


def normalise_query(query: str) -> str:
    """
    Normalise user/evaluation query before retrieval.
    """
    return normalise_text(
        query,
        lowercase=True,
        remove_punctuation=False,
        normalise_whitespace=True,
    )


def chunk_text_for_indexing(chunk: dict) -> str:
    title = chunk.get("title", "")
    section = chunk.get("section", "")
    doc_id = chunk.get("doc_id", "")
    text = chunk.get("text", "")

    weighted_text = f"""
    {doc_id}
    {title}
    {title}
    {section}
    {section}
    {text}
    """

    return normalise_text(
        weighted_text,
        lowercase=True,
        remove_punctuation=False,
        normalise_whitespace=True,
    )

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}