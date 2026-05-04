# Clinical RAG Evaluation Dataset (Synthetic, Evaluation-Driven)

## Overview

This repository includes a **synthetic clinical guideline dataset** designed specifically for **evaluation-driven Retrieval-Augmented Generation (RAG)** systems under real-world constraints.

The dataset is not intended for medical use. It is engineered to support:

- retrieval benchmarking (BM25, dense, hybrid)
- grounded generation evaluation
- hallucination detection
- citation correctness
- abstention behaviour ("I do not know")

Unlike typical demo datasets, this corpus is **deliberately structured to expose model weaknesses**, not just showcase capabilities.



## Dataset Philosophy

Most RAG demos fail because:

- documents are too clean
- answers are trivial to retrieve
- no ambiguity or failure cases exist

This dataset is designed differently:

> It acts as a **controlled stress-test environment** for LLM-based systems.



## Dataset Composition

- **Total documents:** 13
- **Type:** Synthetic clinical guideline-style documents
- **Length:** ~400–900 words per document
- **Structure:** Standardised clinical sections

Each document follows:

- Background
- Population
- Intervention
- Monitoring
- Outcomes
- Risks
- Evidence Summary



## Key Dataset Properties

### 1. Structural Diversity

The dataset includes multiple document types:

| Type | Purpose |
||--|
| Chronic management | baseline RAG evaluation |
| Acute intervention | short-context reasoning |
| Preventive guidelines | long-horizon logic |
| Dose adjustment | iterative reasoning |
| Event-driven monitoring | non-linear retrieval |
| Discontinuation protocols | temporal reasoning |

This ensures models cannot rely on a single reasoning pattern.



### 2. Retrieval Difficulty (Non-Trivial)

The dataset introduces **controlled lexical variability**:

Examples:
- "30% reduction" ↔ "one-third reduction"
- "6 weeks" ↔ "1.5 months"

This forces:

- semantic retrieval (dense embeddings)
- robust ranking (hybrid retrieval)
- evaluation beyond keyword matching



### 3. Noise and Distractors

To simulate real-world documents, selected files include:

- irrelevant epidemiology sections
- repeated sentences
- misleading but similar statements

Example patterns:
- unrelated monitoring schedules
- duplicated intervention rules
- extra background paragraphs

This prevents trivial retrieval success.



### 4. Contradictory Evidence

The dataset includes explicit **conflict scenarios**:

- different study populations
- conflicting endpoints (symptoms vs biomarkers)
- inconsistent timeframes

Purpose:
- test verification pipelines
- evaluate groundedness
- prevent overconfident generation



### 5. Ambiguity and Uncertainty

Some documents contain:

- undefined terms ("clinically meaningful")
- conflicting thresholds (10%, 15%, 20%, 25%)
- inconsistent monitoring recommendations

Purpose:
- test uncertainty handling
- evaluate model calibration
- expose hallucination risk



### 6. Abstention Scenarios

The dataset includes **insufficient evidence cases**:

- missing outcomes
- missing monitoring guidance
- lack of controlled studies

Expected system behaviour:
> "The available evidence is insufficient to answer this question."

This enables evaluation of:
- abstention accuracy
- false confidence rate
- safety behaviour



### 7. Internal Inconsistencies

Some documents contain **minor contradictions within the same text**, such as:

- differing outcome timelines
- conflicting monitoring interpretations

Purpose:
- test claim-level verification
- stress evaluation pipelines
- require reasoning beyond surface reading



### 8. Atomic Fact Structure

All documents are written using:

- short, factual sentences
- explicit numerical thresholds
- clearly defined outcomes

This enables:
- claim extraction
- groundedness scoring
- hallucination detection (FActScore-style)



## Evaluation Use Cases

This dataset is designed to support:

### Retrieval Evaluation
- Recall@k
- MRR
- nDCG
- Precision@k

### Generation Evaluation
- Answer correctness
- Groundedness (faithfulness)
- Citation precision / recall

### Hallucination Evaluation
- Unsupported claim rate
- Hallucination rate
- False confidence rate

### Safety & Reliability
- Abstention accuracy
- Over-refusal / under-refusal
- Clinical risk handling



## Why Synthetic?

Synthetic data was chosen to:

- ensure full control over ground truth
- enable reproducibility
- avoid licensing and privacy issues
- explicitly design failure cases

The dataset is intentionally **small but high-quality**, prioritising:

> evaluation signal > dataset size



## Limitations

- Not representative of full clinical complexity
- Limited linguistic variability compared to real-world corpora
- Requires extension for large-scale benchmarking



## Intended Usage

This dataset is intended for:

- research prototyping
- evaluation pipeline development
- RAG system benchmarking
- portfolio demonstration

It is **not intended for medical decision-making**.



## Next Steps in the Repository

This dataset feeds directly into:

- chunking experiments
- hybrid retrieval (BM25 + dense)
- reranking strategies
- agentic verification workflows
- evaluation framework



## Summary

This dataset provides a **controlled, evaluation-driven testbed** for:

- retrieval quality
- grounded generation
- hallucination resistance
- safe abstention behaviour

It is designed to demonstrate **senior-level system thinking**, not just model usage.