from src.generation.rag_pipeline import RAGPipeline
from src.generation.local_llm import MockLLM
from src.retrieval.schemas import RetrievalResult

class DummyRetriever:
    def retrieve(self, query, top_k=10):
        return [
            RetrievalResult(
                chunk_id="doc_001_monitoring_chunk_001",
                doc_id="doc_001",
                title="Test",
                section="Monitoring",
                text="Follow-up monitoring is recommended within 2–4 weeks.",
                score=1.0,
                rank=1,
                retrieval_method="hybrid",
                metadata={},
            )
        ]

pipeline = RAGPipeline(DummyRetriever(), MockLLM())
response = pipeline.answer("What monitoring is recommended?")
print(response.to_dict())
