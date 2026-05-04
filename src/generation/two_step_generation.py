from typing import List, Dict, Any

from src.generation.extractive import build_extractive_answer
from src.generation.prompts import build_rewrite_prompt
from src.generation.llm import LocalLLM
from src.generation.verification import verify_answer


class TwoStepGenerator:
    """
    Production-grade 2-step generation:
    Step 1: Deterministic extraction (high precision)
    Step 2: LLM rewrite (fluency + synthesis)

    Safety:
    - Must preserve citations
    - Must pass verification
    - Fallback to extractive if unsafe
    """

    def __init__(self, llm: LocalLLM, config: Dict[str, Any]):
        self.llm = llm
        self.config = config

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Dict[str, Any]:

        
        # STEP 1 — Extractive Answer
        
        extractive = build_extractive_answer(query, retrieved_chunks)

        if not extractive["answer"]:
            return {
                "answer": "I do not know.",
                "citations": [],
                "mode": "abstain"
            }

        
        # STEP 2 — LLM Rewrite
        
        prompt = build_rewrite_prompt(
            query=query,
            extractive_answer=extractive["answer"],
            citations=extractive["citations"],
            chunks=retrieved_chunks
        )

        rewritten = self.llm.generate(prompt)

        
        # POST-PROCESSING
        
        rewritten_answer = rewritten.strip()

        
        # VERIFICATION LAYER
        
        is_valid = verify_answer(
            answer=rewritten_answer,
            chunks=retrieved_chunks,
            required_citations=extractive["citations"]
        )

        if not is_valid:
            # Fallback to safe answer
            return {
                "answer": extractive["answer"],
                "citations": extractive["citations"],
                "mode": "fallback_extractive"
            }

        return {
            "answer": rewritten_answer,
            "citations": extractive["citations"],
            "mode": "two_step"
        }