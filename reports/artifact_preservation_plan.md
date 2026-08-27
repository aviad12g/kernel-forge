# OpenKernelForge Artifact Preservation Plan

This plan defines the research artifacts that should be copied from RunPod into the local workspace or another durable storage location. It excludes secrets, model weights, Hugging Face caches, `.env` files, vLLM server logs unless explicitly sanitized, `__pycache__`, and `.git` metadata.

Suggested local layout:

```text
artifacts/
  runpod_imports/
    artifact_manifest.json
    SHA256SUMS
    runs/
    20260520_155839_template_fused8_rigorous/
    20260520_163344_gemini_fused8_rigorous/
    20260520_163607_openai_mini_fused8_rigorous/
    20260519_213349_template_fused8_wide/
    20260519_215314_gemini_fused8_baseline/
    20260519_215439_gemini_fused8_template_guided/
    20260520_083300_openai_mini_fused8_cheap/
    20260520_085334_openai_gpt55_fused8_cheap/
    20260520_114551_qwen7b_fused8_cheap/
    datasets/
    fused8_curated_v1/
    reports/
    rigorous_fused8_model_comparison.md
    fused8_phase11_conclusion.md
    fused8_repeatability_comparison.md
    fused8_gemini_vs_template_comparison.md
    fused8_all_model_comparison.md
```

## Required Run Artifacts

| Artifact | RunPod source path | Intended local path | Required | Why it matters | Expected key files |
| --- | --- | --- | --- | --- | --- |
| Rigorous deterministic fused8 template | `/workspace/openkernelforge/runs/20260520_155839` | `artifacts/runs/20260520_155839_template_fused8_rigorous/` | yes | Current paper-facing deterministic template baseline using CUDA-event timing. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `summary.md`, `fused8_report.md`, `repeatability_results.json`, `repeatability_report.md`, `candidates/`, `logs/` |
| Rigorous Gemini fused8 baseline | `/workspace/openkernelforge/runs/20260520_163344` | `artifacts/runs/20260520_163344_gemini_fused8_rigorous/` | yes | Current paper-facing Gemini comparison under rigorous timing. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `repeatability_results.json`, `repeatability_report.md`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| Rigorous OpenAI mini fused8 baseline | `/workspace/openkernelforge/runs/20260520_163607` | `artifacts/runs/20260520_163607_openai_mini_fused8_rigorous/` | yes | Current paper-facing OpenAI mini comparison under rigorous timing. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `repeatability_results.json`, `repeatability_report.md`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| Deterministic fused8 template wide | `/workspace/openkernelforge/runs/20260519_213349` | `artifacts/runs/20260519_213349_template_fused8_wide/` | yes | Establishes deterministic template performance floor. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `summary.md`, `fused8_report.md`, `repeatability_results.json`, `repeatability_report.md`, `candidates/`, `logs/` |
| Gemini fused8 baseline | `/workspace/openkernelforge/runs/20260519_215314` | `artifacts/runs/20260519_215314_gemini_fused8_baseline/` | yes | Main Gemini zero-shot fused8 comparison. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `repeatability_results.json`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| Gemini fused8 template-guided | `/workspace/openkernelforge/runs/20260519_215439` | `artifacts/runs/20260519_215439_gemini_fused8_template_guided/` | yes | Measures whether template context helps and supplies optimization examples. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `performance_search_report.md`, `repeatability_results.json`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| OpenAI mini cheap | `/workspace/openkernelforge/runs/20260520_083300` | `artifacts/runs/20260520_083300_openai_mini_fused8_cheap/` | yes | Cheap OpenAI reference run under the same fused8 protocol. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `repeatability_results.json`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| GPT-5.5 cheap | `/workspace/openkernelforge/runs/20260520_085334` | `artifacts/runs/20260520_085334_openai_gpt55_fused8_cheap/` | yes | Stronger OpenAI cheap baseline; useful comparison even though it was not dominant. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `real_run_review.md`, `fused8_report.md`, `repeatability_results.json`, `candidates/`, `prompts/`, `responses/`, `logs/` |
| Qwen 7B cheap | `/workspace/openkernelforge/runs/20260520_114551` | `artifacts/runs/20260520_114551_qwen7b_fused8_cheap/` | yes | Local/open model negative baseline; important for honest comparison. | `config.yaml`, `environment_probe.json`, `results.jsonl`, `analysis.md`, `fused8_report.md`, `candidates/`, `prompts/`, `responses/`, `logs/` |

## Required Dataset Artifact

| Artifact | RunPod source path | Intended local path | Required | Why it matters | Expected key files |
| --- | --- | --- | --- | --- | --- |
| Curated fused8 dataset | `/workspace/openkernelforge/datasets/fused8_curated_v1` | `artifacts/datasets/fused8_curated_v1/` | yes | Repeatability-aware dataset for future SFT/optimization work after manual review. | `manifest.json`, `README.md`, `correct_fast_repeat_stable.jsonl`, `correct_fast_single_run.jsonl`, `correct_promising.jsonl`, `optimization_pairs_template_vs_gemini.jsonl`, `optimization_pairs_gemini_vs_template.jsonl`, `rejected_or_unstable.jsonl` |

## Required Final Reports

| Artifact | RunPod source path | Intended local path | Required | Why it matters | Expected key files |
| --- | --- | --- | --- | --- | --- |
| Rigorous fused8 model comparison | `/workspace/openkernelforge/runs/rigorous_fused8_model_comparison.md` | `artifacts/reports/rigorous_fused8_model_comparison.md` | yes | Current paper-facing comparison of templates, Gemini, and OpenAI mini under rigorous timing. | Markdown report |
| Final fused8 conclusion | `/workspace/openkernelforge/runs/fused8_phase11_conclusion.md` | `artifacts/reports/fused8_phase11_conclusion.md` | yes | Final fused8 interpretation. | Markdown report |
| Repeatability comparison | `/workspace/openkernelforge/runs/fused8_repeatability_comparison.md` | `artifacts/reports/fused8_repeatability_comparison.md` | yes | Stable winner comparison across template/Gemini runs. | Markdown report |
| Gemini vs template comparison | `/workspace/openkernelforge/runs/fused8_gemini_vs_template_comparison.md` | `artifacts/reports/fused8_gemini_vs_template_comparison.md` | optional | Useful if present for model-vs-template summary. | Markdown report |
| All-model comparison | `/workspace/openkernelforge/runs/fused8_all_model_comparison.md` | `artifacts/reports/fused8_all_model_comparison.md` | optional | Useful if present for final comparison across all model families. | Markdown report |

## Packaging Command On RunPod

```bash
cd /workspace/openkernelforge
python scripts/package_runpod_artifacts.py \
  --out openkernelforge_fused8_artifacts.tar.gz
```

The package script records missing paths in `artifact_manifest.json` and continues. It writes `SHA256SUMS` for import-time validation.

## Local Import And Validation

```bash
python scripts/import_runpod_artifacts.py \
  --source-root /workspace/openkernelforge \
  --out-dir artifacts/runpod_imports

python scripts/validate_research_package.py
python scripts/update_artifact_index.py
```

If the RunPod workspace is not mounted locally, the import script records missing runs in `artifacts/runpod_imports/artifact_manifest.json`. In the current local package, the KernelBench pilot and repair artifacts are imported, while the three rigorous fused8 run directories remain summarized only.

## Safety Rules

- Do not copy API keys, `.env` files, model weights, Hugging Face caches, or vLLM server logs unless logs are manually sanitized.
- Do not treat Qwen 14B as a model result; the failure was disk/cache capacity during serving.
- Do not publish single-run wins as stable benchmark claims without repeatability.
- This is an internal fused8 artifact package, not KernelBench and not a SOTA claim.
