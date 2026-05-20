# OpenKernelForge Public/Research Review Checklist

Use this checklist before publishing the repository, sharing a report bundle, or starting the next benchmark phase.

- [ ] `pytest -q` passes.
- [ ] `python scripts/check_research_artifacts.py` passes.
- [ ] `python scripts/validate_research_package.py` passes, or missing artifacts are explicitly documented.
- [ ] No secrets are present in reports, run artifacts, configs, datasets, or logs.
- [ ] README has been reviewed for accuracy.
- [ ] `reports/openkernelforge_technical_report.md` has been reviewed.
- [ ] Dataset README is present for any imported curated dataset.
- [ ] Run artifacts are imported under `artifacts/` or explicitly marked missing in `reports/artifact_index.md`.
- [ ] The report makes no SOTA claim.
- [ ] Limitations are stated clearly.
- [ ] License file is present or marked TODO before public release.
- [ ] Citation/BibTeX entry is TODO if a formal artifact release is planned.
- [ ] Decide whether to publish full run artifacts, only reports, or a reduced sanitized artifact bundle.

Current release posture: suitable for internal research review after artifact import and validation. Not yet a KernelBench result, not a trained model release, and not a SOTA claim.
