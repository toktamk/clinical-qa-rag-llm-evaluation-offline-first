from src.generation.extractive_answerer import ExtractiveAnswerer

chunks = [
    {
        "chunk_id": "doc_001_monitoring_chunk_001",
        "text": "Patients starting Therapy A should undergo baseline blood testing. Follow-up blood monitoring is recommended within 2–4 weeks after initiation and then every 3 months."
    }
]

answer = ExtractiveAnswerer().answer(
    question="What monitoring is recommended after Therapy A?",
    retrieved_chunks=chunks,
)

print(answer.to_dict())
