🧠 End-to-End Workflow: Evaluation-Driven Clinical RAG System
This project follows a strict evaluation-first pipeline, not a typical “build chatbot → test later” approach.
The central principle is:

```text
Evaluation comes BEFORE modelling
```

The work completed so far establishes a complete benchmark foundation for an offline-first clinical RAG system: synthetic documents, chunking, QA generation, validation, sparse retrieval, dense retrieval, hybrid retrieval, and retrieval benchmarking.

## System Summary (TL;DR)

This project implements a fully offline, evaluation-driven clinical RAG system with:

* synthetic, traceable QA benchmark dataset
* hybrid retrieval (BM25 + dense, weighted fusion)
* citation-grounded generation pipeline
* deterministic verification layer
* evaluation framework covering grounding, citation quality, abstention, and latency

Key result:

> Weighted hybrid retrieval + extractive generation provides the best reliability–latency trade-off, while two-step generation improves readability at significant computational cost.

🧩 Phase 1 — Problem Framing \& System Design
Objective
Design a production-grade, offline-first clinical RAG system that demonstrates:
retrieval quality
grounded generation
hallucination resistance
safe abstention behaviour
reproducible local execution
Key design decision

```text
Evaluation-first development
```

Instead of building a chatbot first, the project first constructs:
a controlled document corpus,
a chunked retrieval corpus,
a QA evaluation dataset,
retrieval benchmarks,
and only then moves towards generation.


📄 Phase 2 — Synthetic Dataset Design
What we did
We created 13 synthetic clinical guideline-style documents with deliberately engineered properties.
Why synthetic?
Synthetic documents were used because they provide:
full control over ground truth
no licensing/privacy risk
deterministic evaluation examples
deliberately designed failure cases
reproducibility for CI and local testing
Dataset properties
Property	Purpose
Structured sections	predictable parsing and chunking
Numerical thresholds	precise QA and retrieval evaluation
Conditional rules	reasoning evaluation
Noise/distractors	retrieval robustness
Repeated content	chunking and deduplication stress test
Contradictions	verifier and groundedness testing
Ambiguity	uncertainty handling
Missing evidence	abstention testing
Clinical-risk language	safety evaluation
Document types included

```text
- chronic management
- acute intervention
- preventive guideline
- dose adjustment
- risk-stratified management
- event-driven monitoring
- discontinuation protocol
- contradictory evidence
- ambiguous evidence
- insufficient evidence
```

This provides broad coverage for retrieval, generation, hallucination, and abstention evaluation.

🔍 Phase 3 — Dataset Validation \& Refinement
What we did
We iteratively reviewed the synthetic documents to ensure they were not redundant, overly similar, or too clean.
Key refinements

```text
\[✓] checked for duplicate and near-duplicate documents
\[✓] reduced repeated chronic-management patterns
\[✓] added event-driven monitoring logic
\[✓] added discontinuation/reinitiation logic
\[✓] added explicit contradictory evidence in doc\_011
\[✓] strengthened ambiguity in doc\_012 using conflicting thresholds
\[✓] added insufficient-evidence case in doc\_013
\[✓] added distractors and irrelevant epidemiology
\[✓] added paraphrasing such as “30%” vs “one-third”
\[✓] added noise, repeated statements, and minor internal tension in doc\_006
```

Outcome

```text
Dataset quality: evaluation-grade synthetic benchmark
```

The corpus is now suitable for stress-testing retrieval, grounding, citation validity, and abstention behaviour.

⚙️ Phase 4 — Chunking Pipeline
Goal
Convert raw synthetic documents into retrieval-ready chunks.
Files implemented

```text
src/chunking/document\_parser.py
src/chunking/section\_chunker.py
src/chunking/chunk\_schema.py
src/cli/build\_chunks.py
```

Pipeline

```text
raw synthetic documents
→ parse document ID, title, and sections
→ normalise headings and section names
→ section-aware chunking
→ sentence-aware splitting for longer sections
→ metadata inference
→ chunk validation
→ export chunks.jsonl
```

Output

```text
data/processed/chunks.jsonl
```

Result

```text
91 chunks generated
```

Chunk schema
Each chunk contains:

```json
{
  "chunk\_id": "doc\_001\_monitoring\_chunk\_001",
  "doc\_id": "doc\_001",
  "title": "Monitoring Strategy for Therapy A in Condition X",
  "section": "Monitoring",
  "text": "...",
  "token\_count": 50,
  "source\_path": "data/raw/synthetic\_demo/doc\_001.txt",
  "metadata": {
    "document\_type": "chronic\_management",
    "contains\_noise": true,
    "contains\_contradiction": false,
    "contains\_ambiguity": false,
    "supports\_abstention": false,
    "contains\_conditional\_rule": true
  }
}
```

Validation outcome
We verified:

```text
\[✓] all source documents parse correctly
\[✓] section mapping works
\[✓] stable chunk IDs are generated
\[✓] metadata flags are attached
\[✓] short chunks are meaningful, especially for ambiguity and abstention cases
\[✓] chunks.jsonl is usable for retrieval and QA mapping
```

Short chunks in doc\_012 and doc\_013 were intentionally retained because they support ambiguity and insufficient-evidence evaluation.

🧪 Phase 5 — QA Generation Pipeline
Goal
Create a traceable evaluation dataset, not just a list of questions.
Every QA item must map back to chunk-level evidence.
File implemented

```text
src/cli/generate\_eval\_qa.py
```

Strategy
We used template-based QA generation for the MVP.
Why template-based generation first?

```text
\[✓] deterministic
\[✓] reproducible
\[✓] offline-compatible
\[✓] easy to validate
\[✓] avoids hallucinated synthetic labels
\[✓] maps cleanly to known chunks
```

QA types implemented

```text
direct\_evidence
multi\_hop\_evidence
clinical\_risk\_sensitive
temporal\_reasoning
citation\_stress\_test
conflicting\_evidence
ambiguous\_evidence
insufficient\_evidence
intervention questions
evidence-summary questions
```

Additional generators added
To increase dataset size from 59 to 100 QA items, we added:

```text
generate\_intervention\_qa()
generate\_evidence\_summary\_qa()
generate\_temporal\_qa()
generate\_citation\_stress\_qa()
```

Output

```text
data/evaluation/clinical\_rag\_eval\_full.draft.jsonl
```

Result

```text
100 draft QA items generated
```

QA schema
Each QA item includes:

```json
{
  "question\_id": "q\_doc\_001\_001",
  "question": "According to the guideline, what monitoring is recommended in doc\_001?",
  "category": "direct\_evidence",
  "difficulty": "easy",
  "gold\_answer\_short": "...",
  "gold\_answer\_long": "...",
  "source\_doc\_ids": \["doc\_001"],
  "relevant\_doc\_ids": \["doc\_001"],
  "relevant\_chunk\_ids": \["doc\_001\_monitoring\_chunk\_001"],
  "required\_citations": \["doc\_001\_monitoring\_chunk\_001"],
  "atomic\_claims": \[...],
  "requires\_abstention": false,
  "clinical\_risk": "medium",
  "review\_status": "draft"
}
```

Key design feature

```text
Answer → evidence chunks → source documents
```

This enables:
retrieval evaluation
groundedness evaluation
citation correctness
hallucination detection
abstention scoring

🔎 Phase 6 — QA Validation Pipeline
Goal
Ensure the QA dataset is structurally valid and usable for benchmark evaluation.
File implemented

```text
src/cli/validate\_eval\_dataset.py
```

Validation checks

```text
\[✓] question\_id is unique
\[✓] required fields exist
\[✓] question format is valid
\[✓] relevant\_chunk\_ids exist in chunks.jsonl
\[✓] required\_citations exist in chunks.jsonl
\[✓] atomic\_claims exist
\[✓] atomic claims reference known chunks
\[✓] category labels are valid
\[✓] review\_status is valid
\[✓] abstention items are correctly labelled
\[✓] answerable items contain citations
```

Output

```text
reports/evaluation\_data\_validation.json
```

Result

```text
Dataset validated with zero critical errors.
Warnings, if present, are treated as review signals rather than pipeline failures.
```

🧱 Phase 7 — Retrieval System Design
Goal
Implement a fully offline retrieval layer over `chunks.jsonl` using:

```text
BM25 sparse retrieval
Dense embedding retrieval
Hybrid retrieval
```

Design rationale
The retrieval system is modular so that BM25, dense retrieval, and hybrid retrieval can be evaluated independently using the same QA dataset.
The shared retriever interface is:

```python
results = retriever.retrieve(query="...", top\_k=10)
```

Each result returns:

```text
chunk\_id
doc\_id
title
section
text
score
rank
retrieval\_method
metadata
```

🔤 Phase 8 — BM25 Sparse Retrieval
Goal
Build a fast lexical baseline for exact entity, section, therapy, condition, and numeric matching.
Files implemented

```text
src/retrieval/schemas.py
src/retrieval/text\_normalisation.py
src/retrieval/bm25\_index.py
```

Activities completed

```text
\[✓] implemented shared RetrievalResult schema
\[✓] implemented IndexedChunk abstraction
\[✓] implemented text normalisation
\[✓] implemented BM25 tokenisation
\[✓] implemented BM25Retriever
\[✓] implemented BM25 index persistence
\[✓] built BM25 index from chunks.jsonl
\[✓] tested BM25 query execution
```

Important optimisation
BM25 initially retrieved generic monitoring chunks above the correct Therapy A chunk.
We fixed this by weighting:

```text
doc\_id
title
title
section
section
chunk text
```

This improved entity-specific retrieval and moved the expected chunk to rank 1 for the Therapy A monitoring query.
Output

```text
data/indexes/bm25/bm25.pkl
```

🧬 Phase 9 — Dense FAISS Retrieval
Goal
Implement semantic retrieval using local embeddings and FAISS.
File implemented

```text
src/retrieval/dense\_index.py
```

Tools used

```text
sentence-transformers
BAAI/bge-small-en-v1.5
FAISS IndexFlatIP
CPU inference
```

Activities completed

```text
\[✓] implemented DenseRetriever
\[✓] encoded chunk text locally
\[✓] normalised embeddings for cosine similarity
\[✓] built FAISS index
\[✓] saved dense index and metadata
\[✓] loaded dense index from disk
\[✓] tested semantic retrieval queries
```

Output

```text
data/indexes/faiss/index.faiss
data/indexes/faiss/metadata.json
```

Observation
Dense retrieval worked technically but was weaker on exact entity-specific queries such as `Therapy A`.
This is expected because dense retrieval emphasises semantic similarity and can underweight exact identifiers.
This finding became the motivation for hybrid retrieval.

🔀 Phase 10 — Hybrid Retrieval with RRF
Goal
Combine BM25 and dense retrieval to capture both lexical precision and semantic similarity.
Files implemented

```text
src/retrieval/rrf.py
src/retrieval/hybrid\_retriever.py
```

Method

```text
BM25 top-k candidates
+
Dense top-k candidates
→ Reciprocal Rank Fusion
→ final ranked results
```

Activities completed

```text
\[✓] implemented Reciprocal Rank Fusion
\[✓] implemented HybridRetriever
\[✓] loaded BM25 and dense indexes together
\[✓] tested hybrid retrieval on Therapy A monitoring query
\[✓] confirmed hybrid retrieval ranked the expected chunk at rank 1 for the manual test query
```

Observation
RRF hybrid worked correctly, but benchmark results showed it did not consistently outperform BM25 across the full dataset.
This motivated weighted hybrid fusion.

⚖️ Phase 11 — Weighted Hybrid Fusion
Goal
Improve hybrid retrieval by giving BM25 stronger influence while retaining dense semantic signal.
File implemented

```text
src/retrieval/weighted\_fusion.py
```

Fusion formula

```text
hybrid\_score = 0.7 × normalised\_bm25\_score + 0.3 × normalised\_dense\_score
```

Activities completed

```text
\[✓] implemented min-max score normalisation
\[✓] implemented weighted score fusion
\[✓] added fusion\_method parameter to HybridRetriever
\[✓] added bm25\_weight and dense\_weight parameters
\[✓] updated retrieval CLI evaluation to support weighted fusion
\[✓] evaluated weighted hybrid against BM25, dense, and RRF hybrid
```

Result
Weighted hybrid became the best overall retriever.

📊 Phase 12 — Retrieval Evaluation Framework
Goal
Evaluate retrieval quality using the generated QA dataset and known relevant chunk IDs.
Files implemented

```text
src/evaluation/retrieval\_metrics.py
src/cli/evaluate\_retrieval.py
```

Metrics implemented

```text
Hit@k
Recall@k
Precision@k
MRR@k
nDCG@k
mean latency
p50 latency
p95 latency
```

Evaluation workflow

```text
clinical\_rag\_eval\_full.draft.jsonl
→ use question as retrieval query
→ retrieve top-k chunks
→ compare retrieved\_chunk\_ids to relevant\_chunk\_ids
→ compute metrics
→ save JSON report
```

Reports generated

```text
experiments/results/retrieval\_bm25.json
experiments/results/retrieval\_dense.json
experiments/results/retrieval\_hybrid.json
experiments/results/retrieval\_hybrid\_weighted.json
```

🏁 Phase 13 — Retrieval Benchmark Results
Benchmark table
Retriever	Recall@5 ↑	Recall@10 ↑	MRR@10 ↑	nDCG@10 ↑	p95 latency ↓
BM25	0.6400	0.8517	0.5176	0.5672	0.29 ms
Dense	0.2600	0.4583	0.1887	0.2435	21.56 ms
Hybrid RRF	0.6350	0.8433	0.4521	0.5285	18.62 ms
Hybrid weighted	0.6908	0.8542	0.5691	0.6030	19.23 ms
Interpretation
Weighted hybrid retrieval achieved the best overall retrieval quality.
Compared with BM25, weighted hybrid improved:

```text
Recall@5:   0.6400 → 0.6908
Recall@10:  0.8517 → 0.8542
MRR@10:     0.5176 → 0.5691
nDCG@10:    0.5672 → 0.6030
```

Dense retrieval alone underperformed because the benchmark contains many entity-specific, section-specific, and numerically grounded questions.
BM25 remained extremely fast and strong, but weighted hybrid provided the best ranking quality by combining sparse lexical matching with dense semantic similarity.
README-ready conclusion

```text
Weighted hybrid retrieval achieved the best overall retrieval performance, improving MRR@10 and nDCG@10 over BM25 while preserving high Recall@10. Dense retrieval alone underperformed because the benchmark contains many entity-specific and numerically grounded questions, highlighting the importance of sparse lexical matching in clinical-style RAG.
```

## Phase 14 — Generation System

The system extends retrieval with a citation-grounded generation pipeline.

Pipeline:

```text
query
→ weighted hybrid retrieval
→ extractive answer
→ optional LLM rewrite
→ citation parsing
→ deterministic verification
→ fallback to extractive answer
```

Generation modes:

* extractive (baseline, deterministic)
* llm (direct generation baseline)
* extract\_then\_rewrite (hybrid pipeline)

Design principle:

> Generation is constrained and grounded, not free-form.

## Phase 15 — Verification Layer

A deterministic verifier enforces safety and grounding constraints.

Checks include:

* cited chunk IDs must exist in retrieved context
* non-abstained answers must include citations
* unsupported claims are flagged
* unsafe or recommendation-style language is detected
* abstention behaviour is validated

Design principle:

> No generated answer is trusted without verification.

## Phase 16 — Generation Evaluation

The system includes a full evaluation pipeline over 100 QA items.

Metrics:

* citation precision / recall
* abstention accuracy
* verification pass rate
* latency (mean / p95)
* LLM-judge quality:

  * clarity
  * completeness
  * fluency

Key finding:

> Extractive generation is the most reliable and efficient baseline, while two-step generation preserves safety but introduces significant latency without clear quality gains.

🧱 ## Updated System State

The system has evolved into a complete, evaluation-driven clinical RAG pipeline with fully offline execution.

```text
Data \& Evaluation Layer
- Structured synthetic clinical corpus (13 documents)
- Retrieval-ready chunk dataset (91 chunks)
- Evaluation QA dataset (100 items)
- QA validation pipeline
- Retrieval evaluation metrics and benchmark reports
- Generation evaluation framework
- LLM-judge quality metrics (clarity, completeness, fluency)
- Failure analysis and benchmark artefacts

Retrieval Layer
- BM25 sparse retriever
- Dense FAISS retriever (local embeddings)
- Hybrid retrieval (RRF)
- Weighted hybrid fusion (best-performing configuration)

Generation Layer
- Citation-grounded generation pipeline
- Generation modes:
  • extractive (deterministic baseline)
  • direct LLM (baseline)
  • extract\_then\_rewrite (hybrid pipeline)

Generation Infrastructure
- Generation schemas
- Context packing module
- Citation-grounded prompt builder
- Citation and abstention parser
- Deterministic answer verifier

LLM \& Pipeline Integration
- Local LLM wrapper (Hugging Face, CPU-compatible)
- MockLLM for deterministic offline testing
- End-to-end RAG pipeline (retrieval → generation → verification)
- Single-question RAG CLI for reproducible experiments

```

🧠 What the completed workflow enables

The system now supports controlled, end-to-end evaluation of retrieval, generation, and system-level behaviour under realistic clinical constraints.

**Retrieval evaluation**

```text
Recall@k
Hit@k
MRR@k
nDCG@k
Precision@k
Latency
```

**Generation evaluation**

```text
answer correctness          → alignment with gold answers
groundedness                → consistency with retrieved evidence
citation precision/recall   → correctness and completeness of attribution
unsupported claim rate      → hallucination detection
false confidence rate       → confident answers without evidence
abstention accuracy         → correct refusal behaviour
verification pass rate      → compliance with safety constraints
```

**System-Level Experiments**

```text
BM25 baseline
Dense retrieval baseline
Hybrid RRF ablation
Weighted hybrid (best-performing)
Extractive vs LLM vs two-step generation
Latency vs quality trade-offs
```

>The system is designed not only to generate answers, but to measure how and why those answers succeed or fail.

**Key Capability**

> The project enables reproducible, evaluation-driven experimentation across the full RAG pipeline, from retrieval quality to grounded generation and safety verification.

This is the core requirement for building production-grade, trustworthy LLM systems, particularly in high-risk domains such as clinical decision support.

This enables rigorous evaluation of grounded generation quality, including:

```text
answer correctness          → alignment with gold reference answers
groundedness                → consistency with retrieved evidence
citation precision/recall   → correctness and completeness of evidence attribution
unsupported claim rate      → detection of hallucinated or ungrounded statements
false confidence rate       → confident answers given without sufficient evidence
abstention accuracy         → correct refusal when evidence is insufficient
verification pass rate      → compliance with deterministic safety and citation checks
```

This shifts evaluation from surface-level fluency to evidence-based correctness, traceability, and safety.

System-Level Experiments

The system supports controlled experimentation across retrieval and generation configurations:

```text
BM25 baseline                     → strong lexical matching baseline
Dense retrieval baseline         → semantic similarity baseline
Hybrid RRF ablation              → rank-based fusion comparison
Weighted hybrid ablation         → score-based fusion (best-performing)
Generation mode comparison       → extractive vs LLM vs two-step
Latency vs quality trade-offs    → production feasibility under local inference
```

This enables:

* systematic ablation studies across retrieval and generation components
* analysis of grounding vs fluency vs latency trade-offs
* evaluation of design choices under offline, resource-constrained settings

The system is designed not just to produce answers, but to measure how and why those answers succeed or fail.

## Next Phase — Productionisation \& Full-System Evaluation

The next phase focuses on transitioning from a research-grade prototype to a production-ready, evaluation-driven clinical RAG system.

```text
✔ FastAPI backend (serving layer)
✔ Streamlit frontend (interactive interface)
✔ Docker containerisation (reproducible deployment)
✔ CI/CD integration (testing and automation)
✔ optional LoRA/PEFT fine-tuning (model adaptation)
✔ semantic (claim-level) verification (beyond structural checks)
✔ latency optimisation for local inference
```

## What the Current System Enables

The system already supports end-to-end experimentation across retrieval and generation:

```text
\[✓] retrieval benchmarking across BM25, dense, and hybrid methods
\[✓] single-query RAG answer generation
\[✓] citation-aware generation scaffolding
\[✓] abstention detection for insufficient evidence
\[✓] deterministic verification of outputs
\[✓] offline debugging using MockLLM
\[✓] reproducible JSON artefact generation
```

However, one critical capability is still missing:

```text
generation evaluation has been implemented and benchmarked; the next step is improving quality and reducing latency.
```

### Immediate Next Step — Generation Evaluation

The highest-priority task is to evaluate the generation pipeline across all 100 QA items.

Target module

src/cli/evaluate\_generation.py

**Goal**

Quantitatively measure whether generated answers are:

* correct
* grounded in evidence
* properly cited
* safely abstaining when required

### Evaluation Workflow

```text
clinical\_rag\_eval\_full.draft.jsonl
→ for each question
→ retrieve evidence (weighted hybrid)
→ generate answer (MockLLM or local LLM)
→ parse citations and abstention
→ run deterministic verification
→ compare against ground truth
→ compute metrics
→ save structured report
```

### Core Metrics

```text
citation\_precision        → correctness of cited evidence
citation\_recall           → completeness of cited evidence
citation\_f1               → overall citation quality
abstention\_accuracy       → correct refusal behaviour
false\_abstention\_rate     → abstaining when answer exists
missed\_abstention\_rate    → answering when should abstain
verification\_pass\_rate    → structural and safety compliance
unsupported\_claim\_rate    → hallucination detection
latency (mean/p50/p95)    → production feasibility
```

### Expected Outputs

```text
experiments/results/generation\_eval\_mock.json
experiments/results/generation\_eval\_qwen\_05b.json
```

**Why This Step Is Critical**

>Without generation evaluation, the system can produce answers but cannot verify whether they are correct, grounded, or safe.

This step transforms the project from:

```text
retrieval benchmark + working RAG prototype
```

into:

```text
fully evaluated, production-grade RAG system
```

### Validation Strategy

Two baseline runs establish evaluation correctness:

1. MockLLM (negative control)
* Always abstains
* Validates pipeline correctness
* Establishes lower-bound performance
2. Local LLM (e.g. Qwen 0.5B)
* Produces real answers
* Enables first generation benchmark
* Exposes grounding and citation failures

## Next Roadmap After Evaluation

```text
1. Analyse generation failure cases
2. Improve prompt constraints and structure
3. Refine abstention policy
4. Implement claim-level (semantic) verification
5. Add benchmark tables to README
6. Build FastAPI serving layer
7. Add Streamlit interface
8. Containerise with Docker
9. Add CI tests and regression checks
10. Explore LoRA/PEFT fine-tuning
```

