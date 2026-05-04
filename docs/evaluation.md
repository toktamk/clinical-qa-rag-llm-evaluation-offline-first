# Evaluation Framework

This project follows an evaluation-first methodology:

> Evaluation comes before modelling.

All system components — retrieval, generation, and verification — are designed to be measurable, reproducible, and benchmarked under controlled conditions.

## Evaluation Dataset

The evaluation dataset consists of **100 synthetic clinical QA items**, designed to support:

- retrieval benchmarking  
- grounded generation evaluation  
- citation correctness  
- hallucination detection  
- abstention behaviour  

Each item includes:

- question  
- gold answer  
- relevant documents  
- relevant chunks  
- required citations  
- abstention label  
- difficulty and category  

Example schema:

```json
{
  "question_id": "q_0001",
  "question": "...",
  "gold_answer": "...",
  "relevant_chunk_ids": ["doc_001_chunk_004"],
  "required_citations": ["doc_001_chunk_004"],
  "requires_abstention": false
}
```

## Retrieval Evaluation

Retrieval is evaluated using:

- Recall@k
- MRR@k
- nDCG@k
- Precision@k
- latency metrics

Weighted hybrid retrieval achieved the best overall performance.

## Generation Evaluation

Generation is evaluated across three systems:

- Extractive
- Direct LLM
- Extract → Rewrite

**Metrics:**

- citation precision / recall
- abstention accuracy
- verification pass rate
- latency

- **LLM-judge quality:**
  - clarity
  - completeness
  - fluency

## Verification

A deterministic verifier ensures:

- all citations are valid
- unsupported claims are flagged
- unsafe outputs are rejected
- abstention behaviour is consistent

## Evaluation Workflow
QA dataset
→ retrieval
→ generation
→ citation parsing
→ verification
→ metric computation
→ JSON report

## Key Principles
The system is not evaluated on fluency alone, but on grounded correctness, citation validity, and safe behaviour.

## Reproducibility
Run full evaluation:
``` text
</Bash>
python -m src.cli.evaluate_generation --eval-file data/evaluation/clinical_rag_eval_full.draft.jsonl --retriever hybrid --fusion-method weighted --generation-mode extract_then_rewrite --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --enable-llm-judge --output experiments/results/generation_eval.json
```
