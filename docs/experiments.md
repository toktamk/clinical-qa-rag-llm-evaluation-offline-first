Evaluation dataset schema

Use JSONL:

{
  "question_id": "q_0001",
  "question": "What does the evidence say about X?",
  "gold_answer": "Answer supported by the source.",
  "relevant_doc_ids": ["doc_001"],
  "relevant_chunk_ids": ["doc_001_chunk_004"],
  "required_citations": ["doc_001_chunk_004"],
  "answer_type": "extractive_grounded",
  "difficulty": "medium",
  "requires_abstention": false,
  "risk_level": "low",
  "category": "answerable_single_evidence"
}

Experiment tables for README and paper-style docs
Table 1 — Retrieval method comparison
Retriever	Recall@5 ↑	Recall@10 ↑	MRR@10 ↑	nDCG@10 ↑	Precision@10 ↑	Latency ms ↓
BM25	TBD	TBD	TBD	TBD	TBD	TBD
Dense only	TBD	TBD	TBD	TBD	TBD	TBD
BM25 + Dense RRF	TBD	TBD	TBD	TBD	TBD	TBD
BM25 + Dense + Reranker	TBD	TBD	TBD	TBD	TBD	TBD
Table 2 — Chunking ablation
Chunking	Chunk size	Overlap	Recall@10 ↑	Citation precision ↑	Hallucination rate ↓
Fixed	256	32	TBD	TBD	TBD
Fixed	512	64	TBD	TBD	TBD
Fixed	1024	128	TBD	TBD	TBD
Section-aware	variable	section	TBD	TBD	TBD
Semantic	variable	semantic	TBD	TBD	TBD
Table 3 — Verification ablation
Variant	Correctness ↑	Groundedness ↑	Unsupported claim rate ↓	Abstention accuracy ↑	p95 latency ↓
RAG only	TBD	TBD	TBD	TBD	TBD
RAG + citations	TBD	TBD	TBD	TBD	TBD
RAG + verifier	TBD	TBD	TBD	TBD	TBD
RAG + verifier + abstention	TBD	TBD	TBD	TBD	TBD