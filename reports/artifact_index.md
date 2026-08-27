# OpenKernelForge Artifact Index

This index is generated from the current workspace. Missing artifacts are not inferred or fabricated.

- Imported artifact root: `artifacts/runpod_imports`
- KernelBench repo path used on RunPod: `/workspace/KernelBench`
- KernelBench commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`

| Artifact | Location | Availability | Evidence status | Required |
| --- | --- | --- | --- | --- |
| rigorous deterministic fused8 template | `artifacts/runpod_imports/runs/20260520_155839` | summarized only | supported or provenance artifact | yes |
| rigorous Gemini fused8 baseline | `artifacts/runpod_imports/runs/20260520_163344` | summarized only | supported or provenance artifact | yes |
| rigorous OpenAI mini fused8 baseline | `artifacts/runpod_imports/runs/20260520_163607` | summarized only | supported or provenance artifact | yes |
| rigorous fused8 model comparison | `artifacts/runpod_imports/runs/rigorous_fused8_model_comparison.md` | present under imported artifacts | supported or provenance artifact | yes |
| deterministic fused8 template wide | `artifacts/runpod_imports/runs/20260519_213349_template_fused8_wide` | summarized only | supported or provenance artifact | yes |
| Gemini fused8 baseline | `artifacts/runpod_imports/runs/20260519_215314_gemini_fused8_baseline` | summarized only | supported or provenance artifact | yes |
| Gemini fused8 template-guided | `artifacts/runpod_imports/runs/20260519_215439_gemini_fused8_template_guided` | summarized only | supported or provenance artifact | yes |
| OpenAI mini cheap | `artifacts/runpod_imports/runs/20260520_083300_openai_mini_fused8_cheap` | summarized only | supported or provenance artifact | yes |
| GPT-5.5 cheap | `artifacts/runpod_imports/runs/20260520_085334_openai_gpt55_fused8_cheap` | summarized only | supported or provenance artifact | yes |
| Qwen 7B local | `artifacts/runpod_imports/runs/20260520_114551_qwen7b_fused8_cheap` | summarized only | supported or provenance artifact | yes |
| curated fused8 dataset | `artifacts/runpod_imports/datasets/fused8_curated_v1` | summarized only | supported or provenance artifact | yes |
| final fused8 conclusion | `artifacts/runpod_imports/reports/fused8_phase11_conclusion.md` | summarized only | supported or provenance artifact | yes |
| repeatability comparison | `artifacts/runpod_imports/reports/fused8_repeatability_comparison.md` | summarized only | supported or provenance artifact | yes |
| Gemini/template comparison | `artifacts/runpod_imports/reports/fused8_gemini_vs_template_comparison.md` | optional missing | supported or provenance artifact | optional |
| all-model comparison | `artifacts/runpod_imports/reports/fused8_all_model_comparison.md` | optional missing | supported or provenance artifact | optional |
| KernelBench safe baseline validation | `runs/20260520_181052` | present in workspace | historical evaluator artifact; provisional | yes |
| KernelBench Gemini candidate pilot | `artifacts/runpod_imports/runs/20260520_202314` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench candidate failure analysis | `artifacts/runpod_imports/runs/20260520_202314/kernelbench_candidate_failure_analysis.md` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench failure taxonomy JSON | `artifacts/runpod_imports/runs/20260520_202314/kernelbench_failure_taxonomy.json` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench repair subset | `artifacts/runpod_imports/runs/20260520_202314/kernelbench_repair_subset.md` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench Gemini repair pass | `artifacts/runpod_imports/runs/20260520_213128` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench repair comparison | `artifacts/runpod_imports/runs/kernelbench_gemini_repair1_comparison.md` | present under imported artifacts | historical evaluator artifact; provisional | yes |
| KernelBench memory-safe selection config | `configs/kernelbench_l1_20task_rigorous_safe.yaml` | present in workspace | historical compatibility config; corrected paper campaign uses workshop protocol | optional |
| KernelBench Gemini pilot config | `configs/kernelbench_l1_20task_gemini_rigorous.yaml` | present in workspace | historical adapter provenance; not used for corrected paper results | optional |
| KernelBench repair config | `configs/kernelbench_l1_20task_gemini_repair1.yaml` | present in workspace | historical parent-run provenance only | yes |
| KernelBench interpretation notes | `reports/kernelbench_interpretation_notes.md` | present in workspace | historical evaluator artifact; provisional | yes |
| KernelBench loss-win static analysis | `reports/kernelbench_loss_win_static_analysis.md` | present in workspace | historical evaluator artifact; provisional | yes |
| KernelBench profiler diagnostic status | `reports/profiling/kernelbench_loss_profiler_summary.md` | present in workspace | historical evaluator artifact; provisional | optional |
| Fused8 artifact recovery notes | `reports/fused8_artifact_recovery_notes.md` | present in workspace | supported or provenance artifact | yes |
| KernelBench adapter audit | `reports/kernelbench_adapter_audit.md` | present in workspace | current static audit | yes |
| KernelBench current-policy re-audit | `reports/tables/kernelbench_historical_policy_reaudit.csv` | present in workspace | current static audit | yes |
| Corrected five-task baseline config | `configs/kernelbench_l1_5task_corrected_rigorous.yaml` | present in workspace | compatibility baseline config; workshop campaign uses stricter holdout protocol | optional |
| Corrected 20-task safe baseline config | `configs/kernelbench_l1_20task_corrected_rigorous_safe.yaml` | present in workspace | compatibility baseline config; workshop campaign uses stricter holdout protocol | optional |
| Corrected CUDA campaign runner | `scripts/run_corrected_cuda_campaign.py` | present in workspace | compatibility baseline orchestration; not the paper campaign | optional |
| Workshop 2026 GPU source bundle | `artifacts/openkernelforge_workshop2026_gpu_bundle_v1_10.tar.gz` | present in workspace | executed source bundle; SHA-256 `bb5fdfbe...4e9b` | yes |
| Workshop 2026 prespecified protocol | `configs/workshop2026_holdout_protocol.yaml` | present in workspace | executed checksum-frozen RQ1/RQ3 design | yes |
| Workshop 2026 multiplicity protocol | `configs/workshop2026_multiplicity_protocol.yaml` | present in workspace | executed separate all-candidate RQ2 design | yes |
| Workshop task-manifest freezer | `scripts/freeze_kernelbench_task_selection.py` | present in workspace | executed against pinned official checkout | yes |
| Excluded-task GPU shakedown | `artifacts/workshop2026/shakedown_excluded_task/shakedown_summary.json` | present in workspace | `PASS`; excluded from paper outcomes | yes |
| Workshop paired CUDA worker | `scripts/benchmark_holdout_worker.py` | present in workspace | executed on Tesla T4; raw block records preserved | yes |
| Workshop promotion analysis | `openkernelforge/reports/holdout_confirmation.py` | present in workspace | completed; 0 screening and 0 confirmed above-margin wins | yes |
| Evaluator calibration controls | `artifacts/workshop2026/evaluator_controls/calibration_validity.json` | present in workspace | `PASS` in 7 processes | yes |
| Isolated lifecycle control | `artifacts/workshop2026/lifecycle_ablation/lifecycle_ablation_summary.json` | present in workspace | `PASS`, 24/24 rows | yes |
| Lifecycle uncertainty table | `reports/tables/workshop2026_lifecycle_uncertainty.csv` | present in workspace | process-row IQR and 20,000-sample task-cluster bootstrap from preserved rows | yes |
| Lifecycle uncertainty report | `reports/workshop2026_lifecycle_uncertainty.md` | present in workspace | derived analysis; no CUDA rerun | yes |
| Formal campaign gate | `artifacts/workshop2026/campaign_validity.json` | present in workspace | `PASS` before screening | yes |
| All-candidate multiplicity results | `artifacts/workshop2026/multiplicity/campaign/` | present in workspace | 4 tasks, 28 confirmation processes, 67-entry ledger | yes |
| Near-threshold v1 calibration | `artifacts/workshop2026/near_threshold_multiplicity/campaign/` | present in workspace | calibration-only grid-design pilot; did not enter primary screening | optional |
| Near-threshold v2 design prior | `artifacts/workshop2026/near_threshold_multiplicity_v3/design_prior/calibration_selection_v2.csv` | present in workspace | calibration summary only; failed the 12-per-task window gate and did not enter primary screening | optional |
| Near-threshold v3 protocol | `configs/workshop2026_near_threshold_multiplicity_v3_protocol.yaml` | present in workspace | executed frozen 8-candidate-per-task stress-test design | yes |
| Near-threshold v3 selected manifest | `artifacts/workshop2026/near_threshold_multiplicity_v3/selected_candidate_manifest.json` | present in workspace | 32 candidates frozen after disjoint calibration and before primary screening | yes |
| Near-threshold v3 results | `artifacts/workshop2026/near_threshold_multiplicity_v3/campaign/` | present in workspace | complete; 12 calibration, 4 screening, 28 confirmation processes; 94-entry ledger | yes |
| Near-threshold paper tables | `reports/tables/workshop2026_near_threshold_multiplicity.csv` and `reports/tables/workshop2026_near_threshold_winners.csv` | present in workspace | derived from checksum-verified primary blocks | yes |
| Corrected main-figure builder | `scripts/make_workshop2026_results_figure.py` | present in workspace | regenerated from completed artifacts | yes |
| Workshop task selection manifest | `artifacts/workshop2026/task_selection_manifest.json` | present in workspace | frozen before generation; 48 selected from 49 feasible | yes |
| Corrected candidate manifest | `artifacts/workshop2026/candidate_manifest.json` | present in workspace | frozen before screening; 144 candidates | yes |
| Holdout confirmation results | `artifacts/workshop2026/holdout_campaign/` | present in workspace | complete; 48 screening and 70 confirmation records; 249-entry ledger | yes |
| Complete GPU evidence checkpoint | `artifacts/colab_checkpoints/okf_checkpoint_gpu_complete_v1_10.tar.gz` | present in workspace | SHA-256 `9174cd67...e02f` | yes |
| Corrected result summary | `reports/workshop2026_corrected_results.md` | present in workspace | traced to checksum-verified JSON/CSV artifacts | yes |
| Corrected candidate failure breakdown | `reports/tables/workshop2026_candidate_failure_breakdown.csv` | present in workspace | derived from frozen screening records; mutually exclusive 144-candidate funnel | yes |
| Corrected compiler-rung summary | `reports/tables/workshop2026_compiler_rung.csv` | present in workspace | 47/48 task baselines; 1/10 frozen winners above compiler margin at screening | yes |
| Derived multiplicity uncertainty | `reports/tables/workshop2026_selection_multiplicity.csv` | present in workspace | post-hoc task bootstrap from preserved all-candidate blocks; frozen campaign ledger unchanged | yes |
| Same-GPU easy-grid control | `artifacts/workshop2026/multiplicity_same_gpu_a4500/` | present in workspace | RTX A4500; 32/32 worker records; checksum ledger verifies | yes |
| Same-GPU RQ2 summary | `reports/tables/workshop2026_same_gpu_rq2_control.csv` | present in workspace | apparent and confirmed rates remain 1.0 through K=20 | yes |
| Fresh-process compiler-rung control | `artifacts/workshop2026/compiler_confirmation_a4500/` | present in workspace | 7/7 RTX A4500 processes; checksum ledger verifies | yes |
| Compiler-rung summary | `reports/tables/workshop2026_compiler_confirmation.csv` | present in workspace | one frozen winner, 2.001x median versus compile; primary eager result unchanged | yes |
| Four-page workshop paper | `paper/workshop2026/openkernelforge_workshop2026.pdf` | present in workspace | strict 4-page main matter; external-review draft | yes |
| Workshop upload artifact | `paper/workshop2026/openkernelforge_workshop2026_submission.pdf` | present in workspace | strict 4-page main matter; prepared artifact, not venue submission | yes |

## Generated Reports

- Technical report: `reports/openkernelforge_technical_report.md`
- Reproducibility guide: `reports/reproducibility.md`
- Paper PDF: `paper/openkernelforge_paper.pdf`
- Artifact preservation plan: `reports/artifact_preservation_plan.md`
- Historical KernelBench adapter audit: `reports/kernelbench_adapter_audit.md`
- Historical candidate policy re-audit: `reports/tables/kernelbench_historical_policy_reaudit.csv`
- Corrected CUDA campaign specification: `docs/methodology/corrected_cuda_campaign.md`
- Vultr deployment handoff: `reports/vultr_deployment_readiness.md`
- Workshop GPU handoff: `reports/workshop2026_gpu_handoff.md`
- Four-page workshop source: `paper/workshop2026/`
- Corrected result summary: `reports/workshop2026_corrected_results.md`
- Corrected summary table: `reports/tables/workshop2026_corrected_summary.csv`
- Candidate failure breakdown: `reports/tables/workshop2026_candidate_failure_breakdown.csv`
- Compiler-rung summary: `reports/tables/workshop2026_compiler_rung.csv`
- Derived multiplicity uncertainty: `reports/tables/workshop2026_selection_multiplicity.csv`
- Lifecycle uncertainty: `reports/tables/workshop2026_lifecycle_uncertainty.csv`
- Same-GPU RQ2 control: `reports/workshop2026_same_gpu_rq2_control.md`
- Compiler-rung confirmation: `reports/workshop2026_compiler_confirmation.md`
- Near-threshold multiplicity summary: `reports/workshop2026_near_threshold_multiplicity.md`
