from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.generation.answer_verifier import verify_answer
from src.generation.citation_parser import (
    extract_citations,
    parse_abstention,
    parse_answer_text,
    parse_declared_abstained,
    parse_declared_abstention_reason,
    parse_declared_citations,
)
from src.generation.context_packer import pack_context
from src.generation.extractive_answerer import ExtractiveAnswerer
from src.generation.prompt_builder import build_generation_prompt
from src.generation.schemas import GeneratedAnswer, RAGResponse


class RAGPipeline:
    """
    End-to-end retrieval-augmented generation pipeline.

    Modes:
    - llm
    - extractive
    - extract_then_rewrite
    """

    VALID_GENERATION_MODES = {"llm", "extractive", "extract_then_rewrite"}

    def __init__(
        self,
        retriever: Any,
        llm: Any | None = None,
        *,
        generation_mode: str = "llm",
        max_context_chunks: int = 5,
        max_chars_per_chunk: int = 1600,
    ):
        if generation_mode not in self.VALID_GENERATION_MODES:
            raise ValueError(
                "generation_mode must be one of: "
                f"{', '.join(sorted(self.VALID_GENERATION_MODES))}"
            )

        if generation_mode in {"llm", "extract_then_rewrite"} and llm is None:
            raise ValueError(
                f"generation_mode='{generation_mode}' requires an llm instance."
            )

        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive.")

        if max_chars_per_chunk <= 0:
            raise ValueError("max_chars_per_chunk must be positive.")

        self.retriever = retriever
        self.llm = llm
        self.generation_mode = generation_mode
        self.extractive_answerer = ExtractiveAnswerer()
        self.max_context_chunks = max_context_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

    def answer(self, question: str, *, top_k: int = 10) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        retrieved_results = self.retriever.retrieve(question, top_k=top_k)

        packed_context = pack_context(
            question=question,
            retrieved_results=retrieved_results,
            max_chunks=self.max_context_chunks,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )

        if self.generation_mode == "extractive":
            return self._answer_extractive(
                question=question,
                packed_context=packed_context,
                top_k=top_k,
            )

        if self.generation_mode == "extract_then_rewrite":
            return self._answer_extract_then_rewrite(
                question=question,
                packed_context=packed_context,
                top_k=top_k,
            )

        return self._answer_llm(
            question=question,
            packed_context=packed_context,
            top_k=top_k,
        )

    def _answer_extractive(
        self,
        *,
        question: str,
        packed_context: Any,
        top_k: int,
    ) -> RAGResponse:
        generated_answer = self.extractive_answerer.answer(
            question=question,
            retrieved_chunks=packed_context.chunks,
        )

        generated_answer = self._with_metadata(
            generated_answer,
            {
                "generation_mode": "extractive",
                "top_k": top_k,
                "max_context_chunks": self.max_context_chunks,
                "has_citations": bool(generated_answer.citations),
                "enforced_abstention": False,
            },
        )

        verification = verify_answer(
            generated_answer,
            used_chunk_ids=packed_context.used_chunk_ids,
        )

        return RAGResponse(
            generated_answer=generated_answer,
            verification=verification,
            retrieved_chunks=packed_context.chunks,
        )

    def _answer_extract_then_rewrite(
        self,
        *,
        question: str,
        packed_context: Any,
        top_k: int,
    ) -> RAGResponse:
        extractive_answer = self.extractive_answerer.answer(
            question=question,
            retrieved_chunks=packed_context.chunks,
        )

        if extractive_answer.abstained or not extractive_answer.citations:
            extractive_answer = self._with_metadata(
                extractive_answer,
                {
                    "generation_mode": "extract_then_rewrite",
                    "rewrite_attempted": False,
                    "rewrite_accepted": False,
                    "fallback_reason": "extractive_abstained_or_missing_citations",
                    "top_k": top_k,
                    "max_context_chunks": self.max_context_chunks,
                    "has_citations": bool(extractive_answer.citations),
                    "enforced_abstention": False,
                },
            )

            verification = verify_answer(
                extractive_answer,
                used_chunk_ids=packed_context.used_chunk_ids,
            )

            return RAGResponse(
                generated_answer=extractive_answer,
                verification=verification,
                retrieved_chunks=packed_context.chunks,
            )

        rewrite_prompt = self._build_rewrite_prompt(
            question=question,
            extractive_answer=extractive_answer.answer,
            citations=extractive_answer.citations,
        )

        raw_rewrite = self.llm.generate(rewrite_prompt)
        rewritten_text = parse_answer_text(raw_rewrite).strip()

        rewrite_citations = sorted(
            set(extract_citations(rewritten_text))
            | set(parse_declared_citations(raw_rewrite))
        )

        required_citations = set(extractive_answer.citations)
        citations_preserved = required_citations.issubset(set(rewrite_citations))

        if not rewritten_text or not citations_preserved:
            fallback_answer = replace(
                extractive_answer,
                raw_model_output=raw_rewrite,
            )
            fallback_answer = self._with_metadata(
                fallback_answer,
                {
                    "generation_mode": "extract_then_rewrite",
                    "rewrite_attempted": True,
                    "rewrite_accepted": False,
                    "fallback_reason": "rewrite_failed_citation_preservation",
                    "required_citations": sorted(required_citations),
                    "rewrite_citations": rewrite_citations,
                    "top_k": top_k,
                    "max_context_chunks": self.max_context_chunks,
                    "has_citations": bool(fallback_answer.citations),
                    "enforced_abstention": False,
                },
            )

            verification = verify_answer(
                fallback_answer,
                used_chunk_ids=packed_context.used_chunk_ids,
            )

            return RAGResponse(
                generated_answer=fallback_answer,
                verification=verification,
                retrieved_chunks=packed_context.chunks,
            )

        generated_answer = GeneratedAnswer(
            question=question,
            answer=rewritten_text,
            citations=rewrite_citations,
            abstained=False,
            abstention_reason=None,
            used_chunk_ids=packed_context.used_chunk_ids,
            raw_model_output=raw_rewrite,
            metadata={
                "generation_mode": "extract_then_rewrite",
                "rewrite_attempted": True,
                "rewrite_accepted": True,
                "source_answer_mode": "extractive",
                "top_k": top_k,
                "max_context_chunks": self.max_context_chunks,
                "prompt_chars": len(rewrite_prompt),
                "raw_output_chars": len(raw_rewrite),
                "has_citations": bool(rewrite_citations),
                "enforced_abstention": False,
            },
        )

        verification = verify_answer(
            generated_answer,
            used_chunk_ids=packed_context.used_chunk_ids,
        )

        if verification.verification_status != "passed":
            fallback_answer = replace(
                extractive_answer,
                raw_model_output=raw_rewrite,
            )
            fallback_answer = self._with_metadata(
                fallback_answer,
                {
                    "generation_mode": "extract_then_rewrite",
                    "rewrite_attempted": True,
                    "rewrite_accepted": False,
                    "fallback_reason": "rewrite_failed_verification",
                    "top_k": top_k,
                    "max_context_chunks": self.max_context_chunks,
                    "has_citations": bool(fallback_answer.citations),
                    "enforced_abstention": False,
                },
            )

            fallback_verification = verify_answer(
                fallback_answer,
                used_chunk_ids=packed_context.used_chunk_ids,
            )

            return RAGResponse(
                generated_answer=fallback_answer,
                verification=fallback_verification,
                retrieved_chunks=packed_context.chunks,
            )

        return RAGResponse(
            generated_answer=generated_answer,
            verification=verification,
            retrieved_chunks=packed_context.chunks,
        )

    def _answer_llm(
        self,
        *,
        question: str,
        packed_context: Any,
        top_k: int,
    ) -> RAGResponse:
        prompt = build_generation_prompt(
            question=question,
            context_text=packed_context.context_text,
        )

        raw_output = self.llm.generate(prompt)
        answer_text = parse_answer_text(raw_output)

        citations = sorted(
            set(extract_citations(answer_text))
            | set(parse_declared_citations(raw_output))
        )

        heuristic_abstained, heuristic_reason = parse_abstention(raw_output)
        declared_abstained = parse_declared_abstained(raw_output)
        declared_reason = parse_declared_abstention_reason(raw_output)

        abstained = (
            declared_abstained
            if declared_abstained is not None
            else heuristic_abstained
        )
        abstention_reason = declared_reason or heuristic_reason

        enforced_abstention = False
        if not citations and not abstained:
            abstained = True
            abstention_reason = "missing_citations"
            enforced_abstention = True

        generated_answer = GeneratedAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            abstained=abstained,
            abstention_reason=abstention_reason,
            used_chunk_ids=packed_context.used_chunk_ids,
            raw_model_output=raw_output,
            metadata={
                "generation_mode": "llm",
                "top_k": top_k,
                "max_context_chunks": self.max_context_chunks,
                "prompt_chars": len(prompt),
                "raw_output_chars": len(raw_output),
                "has_citations": bool(citations),
                "enforced_abstention": enforced_abstention,
            },
        )

        verification = verify_answer(
            generated_answer,
            used_chunk_ids=packed_context.used_chunk_ids,
        )

        return RAGResponse(
            generated_answer=generated_answer,
            verification=verification,
            retrieved_chunks=packed_context.chunks,
        )

    @staticmethod
    def _with_metadata(
        answer: GeneratedAnswer,
        metadata_updates: dict[str, Any],
    ) -> GeneratedAnswer:
        return replace(
            answer,
            metadata={
                **(answer.metadata or {}),
                **metadata_updates,
            },
        )

    @staticmethod
    def _build_rewrite_prompt(
        *,
        question: str,
        extractive_answer: str,
        citations: list[str],
    ) -> str:
        citation_text = ", ".join(f"[{citation}]" for citation in citations)

        return f"""
You are rewriting a citation-grounded clinical RAG answer.

Use ONLY the extracted answer below.
Do not add new medical facts.
Do not use external knowledge.
Preserve every citation exactly.
Every factual sentence must include one or more citations.
If you cannot preserve the citations, return the extracted answer unchanged.

Question:
{question}

Extracted answer:
{extractive_answer}

Required citations:
{citation_text}

Return format:
Answer: <rewritten answer with citations>
Citations: <comma-separated citation IDs>
Abstained: false
Abstention reason: null
""".strip()