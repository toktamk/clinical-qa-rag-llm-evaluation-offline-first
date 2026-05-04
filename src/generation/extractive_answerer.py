from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.generation.schemas import GeneratedAnswer


@dataclass(frozen=True)
class ExtractiveAnswerConfig:
    max_sentences: int = 4
    min_sentence_chars: int = 20
    abstain_if_no_overlap: bool = True


class ExtractiveAnswerer:
    """
    Deterministic extractive answer generator.

    This is a strong offline baseline:
    - no LLM required
    - cites every extracted sentence
    - low latency
    - useful when small local LLMs fail citation formatting

    It selects relevant sentences from retrieved chunks and appends chunk citations.
    """

    def __init__(self, config: ExtractiveAnswerConfig | None = None):
        self.config = config or ExtractiveAnswerConfig()

    @staticmethod
    def is_low_information_text(text: str) -> bool:
        lowered = text.lower()

        weak_patterns = [
            "no evidence",
            "no data",
            "not available",
            "unknown",
            "uncertain",
            "not established",
            "poorly studied",
            "inconsistent",
            "variable",
            "not standardised",
        ]

        return any(p in lowered for p in weak_patterns)

    def answer(
        self,
        *,
        question: str,
        retrieved_chunks: list[Any],
    ) -> GeneratedAnswer:
        if not question or not question.strip():
            raise ValueError("question must be non-empty.")

        if not retrieved_chunks:
            return self._abstain(
                question=question,
                used_chunk_ids=[],
                reason="insufficient_evidence",
                raw_model_output="No retrieved chunks were provided.",
            )

        question_terms = normalise_terms(question)

        used_chunk_ids = [get_chunk_id(chunk) for chunk in retrieved_chunks]

        selected: list[tuple[str, str, float]] = []

        low_information_chunks = 0

        for chunk in retrieved_chunks:
            chunk_id = get_chunk_id(chunk)
            text = get_chunk_text(chunk)

            if self.is_low_information_text(text):
                low_information_chunks += 1

            for sentence in split_sentences(text):
                sentence = sentence.strip()

                if len(sentence) < self.config.min_sentence_chars:
                    continue

                score = sentence_overlap_score(
                    sentence=sentence,
                    question_terms=question_terms,
                )

                if score >= 0.1:
                    selected.append((sentence, chunk_id, score))

        # Sort globally
        selected = sorted(selected, key=lambda item: item[2], reverse=True)

        # --- NEW: enforce chunk diversity ---
        final_selected = []
        seen_chunks = set()

        for sentence, chunk_id, score in selected:
            if chunk_id not in seen_chunks:
                final_selected.append((sentence, chunk_id, score))
                seen_chunks.add(chunk_id)

            if len(final_selected) >= self.config.max_sentences:
                break

        # If still not enough, fill remaining
        if len(final_selected) < self.config.max_sentences:
            for item in selected:
                if item not in final_selected:
                    final_selected.append(item)
                if len(final_selected) >= self.config.max_sentences:
                    break

        selected = final_selected

        selected = selected[: self.config.max_sentences]

        # --- NEW: abstain if most evidence is weak ---
        if low_information_chunks >= int(0.7 * len(retrieved_chunks)):
            return self._abstain(
                question=question,
                used_chunk_ids=used_chunk_ids,
                reason="insufficient_evidence",
                raw_model_output="Majority of retrieved evidence is low-information.",
            )

        if not selected and self.config.abstain_if_no_overlap:
            return self._abstain(
                question=question,
                used_chunk_ids=used_chunk_ids,
                reason="insufficient_evidence",
                raw_model_output="No evidence sentences overlapped with the question.",
            )

        if not selected:
            # Conservative fallback: cite first retrieved chunk but do not over-answer.
            first_chunk_id = used_chunk_ids[0]
            answer_text = (
                "The retrieved evidence may be relevant, but no specific supporting "
                f"sentence could be selected [{first_chunk_id}]."
            )

            return GeneratedAnswer(
                question=question,
                answer=answer_text,
                citations=[first_chunk_id],
                abstained=False,
                abstention_reason=None,
                used_chunk_ids=used_chunk_ids,
                raw_model_output=answer_text,
                metadata={
                    "generation_method": "extractive",
                    "selected_sentence_count": 0,
                    "fallback_used": True,
                },
            )

        answer_sentences = [
            f"- {sentence} [{chunk_id}]"
            for sentence, chunk_id, _score in selected
        ]

        answer_text = "\n".join(answer_sentences)

        citations = sorted({chunk_id for _sentence, chunk_id, _score in selected})

        return GeneratedAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            abstained=False,
            abstention_reason=None,
            used_chunk_ids=used_chunk_ids,
            raw_model_output=answer_text,
            metadata={
                "generation_method": "extractive",
                "selected_sentence_count": len(selected),
                "fallback_used": False,
            },
        )

    def _abstain(
        self,
        *,
        question: str,
        used_chunk_ids: list[str],
        reason: str,
        raw_model_output: str,
    ) -> GeneratedAnswer:
        answer_text = "The provided evidence is insufficient to answer this question."

        return GeneratedAnswer(
            question=question,
            answer=answer_text,
            citations=[],
            abstained=True,
            abstention_reason=reason,
            used_chunk_ids=used_chunk_ids,
            raw_model_output=raw_model_output,
            metadata={
                "generation_method": "extractive",
                "selected_sentence_count": 0,
                "fallback_used": False,
            },
        )


def get_chunk_id(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return chunk["chunk_id"]

    return chunk.chunk_id


def get_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return chunk["text"]

    return chunk.text


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter.

    Good enough for synthetic guideline text and avoids extra dependencies.
    """
    if not text:
        return []

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalise_terms(text: str) -> set[str]:
    """
    Convert text into retrieval-style terms.

    Keeps alphanumeric tokens and removes very common low-signal words.
    """
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)

    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    }


def sentence_overlap_score(sentence: str, question_terms: set[str]) -> float:
    sentence_terms = normalise_terms(sentence)

    if not sentence_terms or not question_terms:
        return 0.0

    overlap = sentence_terms & question_terms
    score = len(overlap) / len(question_terms)

    # --- NEW: minimum overlap threshold ---
    if score < 0.1:
        return 0.0

    lowered = sentence.lower()

    answer_signal_terms = [
        "recommended",
        "monitoring",
        "follow-up",
        "baseline",
        "risk",
        "adverse",
        "outcome",
        "defined",
        "evidence",
        "dose",
        "initiated",
        "increase",
        "decrease",
        "weeks",
        "months",
        "%",
    ]

    for term in answer_signal_terms:
        if term in lowered:
            score += 0.05

    return score

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
    "how",
    "in",
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
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}