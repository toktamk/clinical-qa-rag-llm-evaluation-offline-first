# tests/test_two_step.py

from src.generation.rag_pipeline import RAGPipeline
from src.generation.local_llm import MockLLM
from src.retrieval.schemas import RetrievalResult


class DummyRetriever:
    def retrieve(self, query, top_k=10):
        return [
            RetrievalResult(
                chunk_id="doc_001_monitoring_chunk_001",
                doc_id="doc_001",
                title="Monitoring Strategy for Therapy A",
                section="Monitoring",
                text="Follow-up blood monitoring is recommended within 2-4 weeks after Therapy A initiation and then every 3 months.",
                score=1.0,
                rank=1,
                retrieval_method="dummy",
                metadata={},
            )
        ]


def test_two_step_generation_mode_runs():
    pipeline = RAGPipeline(
        retriever=DummyRetriever(),
        llm=MockLLM(),
        generation_mode="extract_then_rewrite",
        max_context_chunks=5,
        max_chars_per_chunk=1600,
    )

    response = pipeline.answer(
        question="What monitoring is recommended after Therapy A?",
        top_k=5,
    )

    result = response.to_dict()

    assert "generated_answer" in result
    assert "verification" in result
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) == 1