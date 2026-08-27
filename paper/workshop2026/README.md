# Workshop 2026 Paper

This directory is the four-page NeurIPS 2026 ML for Systems workshop paper. The
existing `paper/overleaf/` project is the long technical report and audit
appendix. The workshop paper reports the completed corrected holdout,
controlled multiplicity study, and evaluator controls.

The official workshop CFP was checked on 2026-08-27:
<https://mlforsystems.org/call_for_papers.html>. It permits up to four pages of
main text, excludes references and an optional appendix from that limit, and
does not require anonymization. The named author and public artifact are
therefore retained. Recheck the CFP before upload in case the policy changes.

The workshop paper uses the official `neurips_2026.sty` downloaded from the
template linked by the workshop CFP. Build with:

```bash
python scripts/build_workshop2026_paper.py
```

The strict build refuses pending-evidence markers and enforces the four-page
main-matter limit:

```bash
python scripts/build_workshop2026_paper.py --submission-ready
```

The upload-only entry point keeps the official style notice and writes a
separate PDF:

```bash
python scripts/build_workshop2026_paper.py --submission-upload
```

The command prepares an upload artifact; it does not perform a venue submission.

The main text must remain at or below four pages. References and appendices are
outside that limit, but reviewers are not required to read the appendix.

The checked-in evidence is under `artifacts/workshop2026/`. Rebuild the
three-panel result figure and formal PDF with:

```bash
python scripts/analyze_selection_multiplicity.py \
  --timing-blocks artifacts/workshop2026/multiplicity/campaign/all_candidate_timing_blocks.csv \
  --output reports/tables/workshop2026_selection_multiplicity.csv
python scripts/analyze_near_threshold_campaign.py
python scripts/analyze_lifecycle_uncertainty.py
python scripts/make_workshop2026_results_figure.py
python scripts/build_workshop2026_paper.py --submission-ready
```

The review output is `openkernelforge_workshop2026.pdf`; a compatibility copy is
retained as `workshop2026_draft.pdf`. The exact upload artifact is
`openkernelforge_workshop2026_submission.pdf` and is produced only by
`--submission-upload`. Building it does not submit it to a venue.

## Completed evidence

- 48 performance-blind selected KernelBench L1 tasks, with three frozen Gemini
  candidates per task.
- Of 141 evaluated candidate records, 77 failed static policy, none failed only
  an output contract, 9 failed numerical or repeat correctness, 28 failed during
  candidate compilation or execution, and 27 passed the full gate. One
  additional task failed before its three candidates entered evaluation because
  the compiler baseline exhausted memory.
- Compiler baselines were available for 47/48 selected tasks. One of the 10
  frozen valid-task winners cleared the 2% compiler margin at screening while
  remaining below eager; confirmation remeasured candidate versus eager only.
- No frozen task winner exceeded eager by the prespecified 2% margin in either
  screening or seven-process confirmation.
- The easy deterministic grid retained apparent and confirmed win rates of 1.0
  for every budget from 1 through 20. In the separately calibrated
  near-threshold stress test, apparent and confirmed rates at `K=8` were 0.75
  and 0.50; one of three apparent winners did not confirm.
- A same-GPU RTX A4500 replication of the easy grid retained 1.0 apparent and
  confirmed rates through `K=20`, removing the easy-vs-near hardware mismatch.
- A separate seven-process RTX A4500 check confirmed the one frozen
  compiler-relative winner at 2.001x versus `torch.compile max-autotune`; its
  primary below-eager result is unchanged.
- The lifecycle ablation measured 1.053 median synchronized-host inflation,
  IQR `[0.998, 1.114]`, and a 1.0001 median enclosing CUDA-event ratio, IQR
  `[0.9919, 1.0019]`. Task-cluster intervals are broad and are reported as
  heterogeneity diagnostics rather than population-wide effects.

These are bounded campaign results, not a full KernelBench or
state-of-the-art claim. Reproduction commands and checksum provenance are in
`reports/workshop2026_gpu_handoff.md` and `reports/artifact_index.md`.
