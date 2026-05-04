# Two-Step Generation Evaluation
## Objective
Evaluate whether a two-step generation strategy can improve answer presentation while preserving citation grounding, abstention behaviour, and verification pass rate.
The evaluated method is:
```text
retrieved chunks
→ deterministic extractive answer
→ local LLM rewrite
→ citation preservation check
→ deterministic verifier
→ fallback to extractive answer if unsafe
```
## Research Question
Can a local LLM improve clarity, completeness, or fluency without reducing citation precision, citation recall, abstention accuracy, or verifier pass rate?

### Compared Systems

| System             | Description |
|--------------------|------------|
| Extractive         | Deterministic sentence selection from retrieved chunks with citations. |
| Direct LLM         | Local LLM generates directly from retrieved context. |
| Extract → Rewrite  | Deterministic extractive answer is rewritten by a local LLM under citation-preservation constraints. |

### Metrics

| Metric | Purpose |
|-------|--------|
| Verification pass rate | Checks whether the answer passes deterministic structural and safety verification. |
| Abstention accuracy | Checks whether the system abstains when the gold label requires abstention. |
| Citation precision | Measures whether predicted citations are valid retrieved chunks. |
| Citation recall | Measures whether required evidence chunks were cited. |
| Clarity | LLM-judge score for how clear the answer is. |
| Completeness | LLM-judge score for how complete the answer is relative to the question and reference answer. |
| Fluency | LLM-judge score for readability and linguistic quality. |
| Overall quality | Mean of clarity, completeness, and fluency. |
| Mean / p95 latency | Measures production feasibility under local inference constraints. |


## Key Findings
- Direct LLM generation is unsafe under strict citation requirements in the current setup.
- Citation precision drops to 0.0500.
- Citation recall drops to 0.0400.
- Abstention accuracy drops to 0.0500.
- Verification pass rate drops to 0.9200.
- Extractive generation is a strong reliability baseline.
- Verification pass rate is 1.0000.
- Citation precision is 0.9600.
- Abstention accuracy is 0.9500.
- Mean latency is only 39 ms.
- Extract-then-rewrite preserves grounding but is expensive.
- Verification, abstention, citation precision, and citation recall match the extractive baseline.
- Clarity and completeness improve slightly.
- Fluency decreases slightly.
- Mean latency increases to approximately 29.8 seconds.
- Two-step generation is safe but not yet cost-effective.
- The current rewrite stage does not improve overall quality relative to extractive generation.
- The added local LLM latency is substantial.

## Interpretation
The two-step system successfully prevents the LLM rewrite from degrading safety because all rewrites must preserve citations and pass deterministic verification. This is a strong production-safety result.
However, the experiment also shows that a generative rewrite stage is not automatically useful. In this configuration, extractive generation already provides strong clarity, completeness, fluency, and grounding. The rewrite step adds latency without a meaningful improvement in overall quality.
The correct product decision for the current system is therefore:
```text
Use extractive generation as the default low-latency reliable mode.
Expose extract_then_rewrite as an optional high-latency mode only when improved presentation quality is demonstrably valuable.
```
## Failure-Aware Design
The pipeline includes fallback logic:
```text
extractive answer
→ rewrite
→ citation preservation check
→ verifier
→ fallback to extractive answer
```
This prevents common LLM rewrite failures:
- removed citations;
- unsupported new facts;
- malformed answer format;
- unsafe recommendation language;
- missing abstention when evidence is insufficient.

## Reproducibility Commands
Evaluate extractive generation:
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
Compare results:
```bash
python scripts/compare_generation_results.py --extractive experiments/results/generation_eval_extractive_judged.json --llm experiments/results/generation_eval_llm_judged.json --two-step experiments/results/generation_eval_extract_then_rewrite_judged.json --output experiments/results/generation_comparison_table.md
```
## Next Experiments
- Testing a smaller quantised model for lower-latency rewriting.
- Evaluating rewrite-only-on-demand policies.
- Adding human review for a subset of quality scores.
- Adding semantic correctness scoring beyond citation matching.
- Comparing direct LLM vs extract_then_rewrite using a larger local model if hardware allows.