# Repeatability Label Specification

This document records the implemented OpenKernelForge repeatability labels as of the current paper package. It describes the current code path, not a proposed stricter statistical rule.

## Inputs

There are two implemented label paths.

The repeatability-report path is implemented in `openkernelforge/reports/repeatability.py`.

- `original_speedup`: the original single-run or run-summary speedup vs PyTorch eager from `benchmark_summary.speedup_vs_eager`.
- `speedup_values`: repeated rebenchmark speedups vs eager collected by calling `benchmark_task` several times.
- `stats`: mean, median, std, min, max, and coefficient of variation over `speedup_values`.
- `stable`: `true` when coefficient of variation over `speedup_values` is `<= 0.10`.

The rigorous benchmark-summary path is implemented in `openkernelforge/harness/benchmarker.py` and consumed by `openkernelforge/reports/kernelbench_l1.py`.

- `session_speedups`: one speedup vs eager for each independent session.
- `speedup_vs_eager`: across-session median when `independent_sessions > 1`, otherwise the first session speedup.
- `stable_above_eager`: `true` only when the median session speedup is `>= 1.0` and every session speedup is `>= stable_session_threshold`.
- `stable_session_threshold`: default `0.98`.

CI, IQR, p25/p75, and bootstrap summaries are reported when present, but they are not currently used to assign labels.

## Paper Artifact Mapping

Imported fused8 summaries in the local paper package use the legacy
repeatability-report labels because the full rigorous fused8 run directories
are not locally preserved with the current session-level label inputs. The
historical KernelBench pilot and repair artifacts contain fields from the
rigorous benchmark-summary labeler through `stable_above_eager`. A later
adapter audit invalidated their task-state and baseline lifecycle, so these
labels document historical control flow rather than supported performance.

The rigorous session labeler is preferred for corrected runs because
it records the session-level parity condition directly in the benchmark
summary and does not require a separate post-hoc repeatability pass. The
historical fused8 rows are not retroactively relabeled: doing so would require
the original per-session speedup vectors, which are not present in the local
artifact package. Reporting both definitions keeps the historical fused8
artifacts auditable rather than silently changing labels.

## Implemented Decision Trees

### Repeatability-report labels

```text
if repeat_median is missing:
    INSUFFICIENT_DATA
else if repeat_median >= 1.0 and cv <= 0.10:
    REPEAT_STABLE_WIN
else if original_speedup is present and original_speedup >= 1.0 and repeat_median < 1.0:
    SINGLE_RUN_ONLY_WIN
else if repeat_median >= 1.0:
    UNSTABLE
else:
    BELOW_EAGER
```

### Rigorous benchmark-summary labels

```text
if policy fails:
    POLICY_FAILED
else if verification fails:
    VERIFICATION_FAILED
else if speedup_vs_eager is missing:
    INSUFFICIENT_DATA or BENCHMARK_FAILED
else if speedup_vs_eager >= 1.0 and stable_above_eager is true:
    REPEAT_STABLE_WIN
else if speedup_vs_eager >= 1.0:
    UNSTABLE
else:
    BELOW_EAGER
```

The `stable_above_eager` field is computed as:

```text
median(session_speedups) >= 1.0
and all(session_speedup >= stable_session_threshold for session_speedup in session_speedups)
```

With the current default, `stable_session_threshold = 0.98`. This permits small below-parity session variation while requiring an above-parity median.

`SINGLE_RUN_ONLY_WIN` is reserved for the legacy path, where a preserved
single-run result can actually be compared with a later repeat median. The
rigorous session path does not have a separate single-run observation; an
above-eager median that fails the session threshold is therefore `UNSTABLE`.

The thresholds are implementation defaults rather than tuned values. The
legacy `cv <= 0.10` threshold is the stability rule used by older repeatability
reports. The rigorous `stable_session_threshold = 0.98` allows small
session-level timing variation while still requiring the across-session median
to be above eager. The paper does not claim a threshold-sensitivity result
unless the relevant per-session speedup vectors are preserved.

## Missing Intervals

If CI/IQR fields are missing, labels are not inferred from them. The code falls back to the median and stability fields described above. Tables should therefore mark missing intervals as `not preserved` rather than implying that CI was used.

If only a median exists and no stability flag exists, the candidate can be summarized as below or promising in reports, but it should not be promoted to a repeat-stable label.

## Examples

- `bias_relu`: original single-run speedup `1.029x`, repeat median `0.976x`. Because the original speedup was above eager and the repeat median fell below eager, the repeatability-report label is `SINGLE_RUN_ONLY_WIN`.
- The historical KernelBench pilot recorded `CrossEntropyLoss` at `1.992x` with `stable_above_eager = true` and labeled it `REPEAT_STABLE_WIN`.
- The historical pilot recorded `Matmul_with_diagonal_matrices` at `0.984x` and labeled it `BELOW_EAGER`.

The KernelBench examples above describe preserved historical artifacts. A
subsequent adapter audit found that the original path reconstructed
KernelBench `Model` objects inside timed calls and did not implement the
official `ModelNew` lifecycle required for every task that defines `Model`.
Parameterized tasks additionally lost access to initialized state. Those
numerical rows remain auditable but require corrected-adapter revalidation
before they are used as paper performance evidence.

## Proposed Stricter Rule

A stricter future rule would require either a speedup confidence interval lower bound above `1.0` or an IQR-constrained session distribution in addition to the implemented median/session-threshold rule. The current paper does not use that stricter rule, and old results have not been relabeled under it.
