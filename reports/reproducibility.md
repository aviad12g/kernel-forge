# OpenKernelForge Reproducibility Guide

This guide reproduces the harness and internal fused8 workflow. Real Triton performance results require a CUDA GPU with Triton installed.

## 1. Install

```bash
python -m pip install -e .
pytest -q
```

## 2. Environment Check

```bash
python -m openkernelforge.cli env-check
```

For true Triton benchmark results, the viability should be `TRITON_EXECUTION_OK`.

## 3. Fake Smoke Run

```bash
python -m openkernelforge.cli smoke
```

Fake and dummy runs are harness checks only. They are not model benchmarks.

## 4. Fused8 Template Quick

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_quick.yaml \
  --out-name template_fused8_gpu_quick
```

## 5. Fused8 Template Wide

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_wide.yaml \
  --out-name template_fused8_gpu_wide
```

## 6. Repeatability

```bash
python -m openkernelforge.cli repeatability-report \
  --run-dir runs/<run> \
  --top-k 3 \
  --repeats 5
```

## 7. Optional Model Runs

Gemini/OpenAI runs require API keys in environment variables only. Do not commit keys.

```bash
export GEMINI_API_KEY=<your-key>
python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_baseline.yaml --out-name gemini_fused8_gpu_baseline
unset GEMINI_API_KEY
```

OpenAI cheap runs use `OPENAI_API_KEY` and should be kept small unless early results justify more spend.

Local vLLM runs use the OpenAI-compatible local server path:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000
python scripts/run_local_model_fused8.py --config configs/qwen_fused8_gpu_baseline_cheap.yaml --out-name qwen_fused8_cheap
```

## 8. Curate Dataset

```bash
python -m openkernelforge.cli curate-fused8-dataset \
  --template-run runs/<template_run> \
  --gemini-run runs/<gemini_run> \
  --template-guided-run runs/<guided_run> \
  --out-dir datasets/fused8_curated_v1
```

## 9. Validate Curated Dataset

```bash
python -m openkernelforge.cli validate-curated-fused8 --dataset-dir datasets/fused8_curated_v1
```

## 10. Build Research Report

```bash
python scripts/build_phase14_report.py
python scripts/check_research_artifacts.py
```

## Notes

- GPU is required for real Triton correctness/performance claims.
- API keys must be environment variables only.
- `runs/` and `datasets/` should be reviewed before using them for training.
- This is an internal fused8 workflow, not KernelBench and not a SOTA claim.
