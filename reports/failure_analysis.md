# 🔍 Failure Analysis — Evaluation-Driven Clinical RAG System

## 🔬 Concrete Failure Examples

### Example 1 — LLM Failure (Missing Citations → Forced Abstention)

Query:
What monitoring is recommended after Therapy A?

Retrieved:
doc_001_monitoring_chunk_001

LLM Output:
Monitoring is recommended within 2–4 weeks and then every 3 months.

Issue:
No citations → forced abstention

Insight:
Small local LLMs fail structured citation outputs even with correct retrieval.

### Example 2 — Extractive Limitation (Multi-hop)

Query:
What conditions require dose escalation and monitoring adjustment?

Expected:
Multiple chunks

Output:
Only one chunk cited

Impact:
Reduced recall

Insight:
Extractive methods struggle with multi-hop reasoning.

### Example 3 — Conservative Abstention

Query:
What are the risks associated with Condition P?

Output:
System abstains

Cause:
Low-information detection

Insight:
Safety-first design reduces coverage.

## 🧠 Key Insight

Real-world RAG systems must balance:
- grounding (citations)
- coverage (recall)
- safety (abstention)

## Two-Step Generation Failure Analysis

### Failure Mode 1 — No Measurable Quality Gain

**Observation**

`extract_then_rewrite` achieved the same citation precision, citation recall, abstention accuracy, and verification pass rate as the extractive baseline.

**Interpretation**

The rewrite step did not improve grounding metrics because the extractive answer was already highly constrained and citation-complete for many questions.

**Impact**

The system gains no measurable grounding benefit from rewriting unless fluency or synthesis quality is evaluated separately.

**Mitigation**

Add LLM-judge quality scoring for clarity, completeness, and fluency.

### Failure Mode 2 — Severe Latency Overhead

**Observation**

Mean latency increased from approximately 22 ms for extractive generation to approximately 30,860 ms for extract-then-rewrite generation.

**Root Cause**

The local LLM rewrite stage dominates runtime.

**Impact**

This configuration is not suitable for low-latency interactive use on CPU-only hardware.

**Mitigation**

Evaluate smaller local models, quantised GGUF models, reduced `max_new_tokens`, and rewrite-only-on-demand policies.

### Failure Mode 3 — Rewrite Adds Risk Without Verification

**Observation**

A rewrite model may remove citations, alter factual wording, or add unsupported clinical claims.

**Current Safeguard**

The pipeline enforces:

```text
extractive answer
→ rewrite
→ citation preservation check
→ deterministic verifier
→ fallback to extractive answer
```

If the rewritten answer removes required citations, introduces unsupported claims, changes the meaning of the extractive answer, or fails verification, the system discards the rewrite and returns the extractive answer instead.

**Impact**

This prevents the LLM rewrite stage from silently degrading answer safety. The LLM is treated as a controlled rewriting layer, not as the source of truth.

**Mitigation**

Keep deterministic verification mandatory for all rewritten answers. In future experiments, add stricter semantic-drift checks and compare rewritten answers against the extractive source answer.

### Failure Mode 4 — Metrics Do Not Fully Capture Fluency Gains

**Observation**

Citation precision, citation recall, abstention accuracy, and verification pass rate remain unchanged between extractive generation and extract-then-rewrite generation.

**Root Cause**

These metrics evaluate grounding and safety, but they do not fully capture presentation quality.

**Impact**

A rewrite may improve readability or synthesis without changing citation metrics.

**Current Safeguard**

The system adds LLM-judge quality metrics:

- clarity
- completeness
- fluency
- overall quality

**Mitigation**

Use LLM-judge scores as supplementary metrics only. Do not treat them as replacements for citation precision, citation recall, abstention accuracy, or deterministic verification.

### Failure Mode 5 — Local LLM Judge Can Be Noisy

**Observation**

The same small local model used for generation is also used as the offline quality judge.

**Root Cause**

Small local LLMs can produce inconsistent or overly generous quality ratings.

**Impact**

Quality scores should be interpreted as approximate presentation-quality signals, not definitive truth.

**Mitigation**

Future work should add human review for a small sample, compare multiple local judges, and report inter-judge agreement where possible.

### Final Failure Analysis Summary

The system’s main failure pattern is not retrieval alone or generation alone. It is the interaction between retrieval quality, generation constraints, citation enforcement, abstention behaviour, and latency.

The strongest current conclusion is:

Extractive generation is the safest and most efficient default mode. Direct local LLM generation is unsafe under strict citation constraints. Two-step generation is safe when protected by citation preservation and verification gates, but its latency cost must be justified by measurable quality gains.
