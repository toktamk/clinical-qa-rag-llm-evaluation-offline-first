# Offline-First Clinical LLM System

Evaluation-Driven RAG with Grounded Generation Under Real-World Constraints

A production-grade, fully offline Retrieval-Augmented Generation (RAG) system for clinical-style documents under real-world constraints: 
privacy, local inference, citation grounding, abstention, reproducibility, and measurable reliability.


This repository is not a chatbot demo. 

It is an evaluation-first LLM systems project designed to demonstrate senior-level capability in retrieval engineering, 
grounded generation, verification, failure analysis, and production-oriented ML evaluation.

> This project is for research and engineering evaluation only. It is not a medical diagnosis system and must not be used for clinical decision-making.

## Core Capabilities

- Offline-first local inference with open-source models.
- Hybrid retrieval: BM25 + dense embeddings + weighted fusion.
- Section-aware clinical-style chunking.
- Citation-grounded generation.
- Deterministic extractive baseline with strong grounding properties
- Two-step generation: extract evidence first, then rewrite with a local LLM.
- Citation preservation and deterministic verification gates.
- Safe abstention when evidence is insufficient.
- Retrieval, generation, latency, and LLM-judge quality evaluation.
- Failure analysis designed around real RAG risks, not only happy-path examples.
- Fully containerised deployment with FastAPI backend and Streamlit interface

## System Architecture

![System Architecture](docs/SystemArchitecture.png)

### Pipeline overview

```text
User Query
→ Hybrid Retrieval
→ Evidence Chunks
→ Generation Pipeline
→ Verification Layer
→ API and Frontend
→ Evaluation Framework
```

## Dataset Design
The default dataset is synthetic and clinical-style. It is intentionally designed for controlled evaluation rather than clinical use.

It includes:
- structured sections such as background, population, intervention, monitoring, outcomes, risks, evidence summary;
- numerical thresholds and temporal statements;
- conditional rules;
- multi-hop evidence cases;
- contradictory evidence;
- ambiguous evidence;
- insufficient-evidence cases requiring abstention;
- distractors and near-relevant content.

This makes the dataset useful for evaluating retrieval robustness, citation grounding, abstention behaviour, and generation failure modes without privacy or redistribution risk.

## Retrieval Benchmark

Weighted hybrid retrieval achieved the strongest overall retrieval quality on the synthetic clinical QA benchmark.

### Results

| Method           | Recall@5 ↑ | Recall@10 ↑ | MRR@10 ↑ | nDCG@10 ↑ | p95 Latency ↓ |
|------------------|-----------:|------------:|---------:|----------:|--------------:|
| BM25             | 0.6400     | 0.8517      | 0.5176   | 0.5672    | ~0.3 ms       |
| Dense            | 0.2600     | 0.4583      | 0.1887   | 0.2435    | ~21 ms        |
| Hybrid (RRF)     | 0.6350     | 0.8433      | 0.4521   | 0.5285    | ~18 ms        |
| Hybrid (Weighted)| **0.6908** | **0.8542**  | **0.5691** | **0.6030** | ~19 ms        |

### Key Findings

- **Dense retrieval underperforms** in this setting due to:
  - entity-specific queries  
  - section-specific references  
  - numerically grounded questions  

- **BM25 provides strong lexical precision**, especially for structured clinical content.

- **Hybrid retrieval improves overall ranking quality** by combining:
  - sparse lexical matching (BM25)
  - dense semantic similarity (embeddings)

- **Weighted fusion outperforms RRF**, indicating that calibrated weighting is more effective than rank-only aggregation.

### Practical Implication

**For clinical RAG systems:**

Hybrid retrieval is not optional — it is required for robust performance.

**Pure dense retrieval is insufficient for:**
- guideline-based QA  
- structured medical documents  
- evidence-grounded generation  

## Generation Benchmark

The generation pipeline was evaluated across three strategies under strict clinical constraints, including citation grounding, abstention correctness, and verification.

### Results

| Method                | Verif ↑ | Abst Acc ↑ | Cit Prec ↑ | Cit Rec ↑ | Clarity ↑ | Completeness ↑ | Fluency ↑ | Quality ↑ | Latency ↓ |
|----------------------|--------:|-----------:|-----------:|----------:|----------:|----------------:|----------:|----------:|-----------:|
| Extractive           | 1.0000  | 0.9500     | 0.9600     | 0.5150    | 3.90      | 4.19            | 3.92      | 4.00      | 39 ms      |
| Direct LLM           | 0.9200  | 0.0500     | 0.0500     | 0.0400    | 3.04      | 3.25            | 3.03      | 3.11      | ~28,383 ms |
| Extract → Rewrite    | 1.0000  | 0.9500     | 0.9600     | 0.5150    | **3.95**  | **4.23**        | 3.83      | 4.00      | ~29,816 ms |

### Key Findings

- **Direct LLM generation fails under strict grounding constraints**:
  - very low citation precision and recall  
  - extremely poor abstention accuracy  
  - multiple verification failures  
  - high hallucination risk  

- **Extractive generation provides a strong baseline**:
  - perfect verification pass rate  
  - high citation precision  
  - correct abstention behaviour  
  - extremely low latency  

- **Two-step generation (Extract → Rewrite)**:
  - preserves extractive grounding guarantees  
  - slightly improves clarity and completeness  
  - maintains high citation quality  
  - introduces significant latency overhead  

- **Latency remains a critical trade-off for local LLM rewriting**

### Practical Implication

For clinical RAG systems:

> Reliable generation requires grounding-first design.

- Pure LLM generation is **unsafe without verification and citation constraints**  
- Extractive generation is **fast, robust, and production-ready**  
- Hybrid generation improves readability but must justify its computational cost  

### Key Interpretation

This project explicitly surfaces the trade-off:

> **Local LLM rewriting can be safely integrated behind verification gates, but it must deliver measurable quality gains to justify its latency overhead.**

This highlights a central challenge in offline clinical LLM systems:

- reliability vs fluency vs latency  

## Two-Step Generation Design
The two-step generation mode is designed to combine deterministic reliability with optional generative readability.
```text
retrieved chunks
→ deterministic extractive answer
→ local LLM rewrite
→ citation preservation check
→ deterministic verifier
→ fallback to extractive answer if unsafe
```
**Safety properties:**
- the rewrite step may not introduce unsupported facts;
- required citations must be preserved;
- every factual answer must remain grounded in retrieved evidence;
- unsafe rewrites fall back to the extractive answer;
- abstention remains available for insufficient evidence.

## Evaluation Framework
- **Retrieval metrics**
  - Hit@k
  - Recall@k
  - Precision@k
  - MRR@k
  - nDCG@k
  - mean, p50, and p95 retrieval latency
- **Generation metrics**
  - verification pass rate
  - citation precision
  - citation recall
  - citation precision on non-abstained answers
  - citation recall on non-abstained answers
- **abstention accuracy**
  - predicted abstention rate
  - forced abstention rate
  - mean, p50, and p95 generation latency
- **LLM-judge quality metrics**

  The local LLM judge scores answer presentation quality only. It does not replace citation, grounding, or correctness checks.
  - clarity
  - completeness
  - fluency
  - overall quality

## Deployment
The system is fully deployable in a local, offline environment.

**Run with Docker**

```text
docker compose -f docker/docker-compose.yml up --build
```
**API Endpoints**

- POST /query
- POST /retrieve
- POST /evaluate
- GET /health

**Frontend**

Streamlit interface provides:

- query input
- retrieved evidence inspection
- generated answer with citations
- verification results
- latency reporting

## Reproducibility
Build chunks:
```bash
python -m src.cli.build_chunks --input-dir data/raw/synthetic_demo --output data/processed/chunks.jsonl
```
Evaluate extractive generation with quality scoring:
```bash
python -m src.cli.evaluate_generation --eval-file data/evaluation/clinical_rag_eval_full.draft.jsonl --retriever hybrid --fusion-method weighted --generation-mode extractive --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --max-new-tokens 128 --enable-llm-judge --output experiments/results/generation_eval_extractive_judged.json
```
Evaluate direct LLM generation:
```bash
python -m src.cli.evaluate_generation --eval-file data/evaluation/clinical_rag_eval_full.draft.jsonl --retriever hybrid --fusion-method weighted --generation-mode llm --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --max-new-tokens 256 --enable-llm-judge --output experiments/results/generation_eval_llm_judged.json
```
Evaluate extract-then-rewrite generation:
```bash
python -m src.cli.evaluate_generation --eval-file data/evaluation/clinical_rag_eval_full.draft.jsonl --retriever hybrid --fusion-method weighted --generation-mode extract_then_rewrite --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --max-new-tokens 256 --enable-llm-judge --output experiments/results/generation_eval_extract_then_rewrite_judged.json
```
Compare generation results:
```bash
python scripts/compare_generation_results.py --extractive experiments/results/generation_eval_extractive_judged.json --llm experiments/results/generation_eval_llm_judged.json --two-step experiments/results/generation_eval_extract_then_rewrite_judged.json --output experiments/results/generation_comparison_table.md
```
## Project Structure
```text
src/
  chunking/
  retrieval/
  generation/
    schemas.py
    context_packer.py
    citation_parser.py
    answer_verifier.py
    local_llm.py
    extractive_answerer.py
    rag_pipeline.py
  evaluation/
    retrieval_metrics.py
    llm_judge.py
  cli/
    evaluate_retrieval.py
    evaluate_generation.py
    generate_answer.py
scripts/
  compare_generation_results.py
data/
  raw/
  processed/
  evaluation/
experiments/
  results/
reports/
  failure_analysis.md
docs/
  two_step_generation_evaluation.md
docker/
  docker-compose.yml
  Dockerfile
frontend/
  streamlit_app.py
```
## Limitations
- The default corpus is synthetic and designed for evaluation, not clinical deployment.
- The system is not a medical device or diagnosis tool.
- Current LLM experiments use a small local model, which limits rewrite quality.
- CPU-only local inference introduces high latency for LLM-based modes.
- The LLM judge is useful for presentation-quality scoring, but it is not a substitute for human evaluation or groundedness checks.

## Next Engineering Tasks
- Add reranker ablation: hybrid retrieval with and without cross-encoder reranking.
- Add retrieval-to-generation error correlation analysis.
- Add optional LoRA/PEFT fine-tuning comparison.
- Do latency optimisation for local inference
- Do extended evaluation with real-world datasets

## Ethical Safeguards
This repository is intended for AI engineering research and portfolio demonstration only. 

It does not provide clinical advice, diagnosis, or treatment recommendations. 

All answer generation is constrained by retrieved evidence, citation checks, and abstention logic.

## Summary
This project demonstrates production-grade thinking for RAG systems under real-world constraints:
- evaluation-first development;
- offline inference;
- hybrid retrieval;
- citation-grounded generation;
- deterministic verification;
- latency and quality benchmarking;
- failure-aware system design.

The strongest current finding is that safe, deterministic generation outperforms direct local LLM generation under strict citation constraints, while two-step rewriting must be justified by measurable quality gains relative to its latency cost.
