# Label Threshold Sensitivity Notes

This analysis uses existing artifacts only. It does not rerun benchmarks.

Rows written: 28.
Rows without preserved per-session speedups: 28.

Result: the local artifact package does not preserve the per-session speedup vectors needed to recompute headline labels at tau values 0.95, 0.97, 0.98, and 0.99. The artifacts preserve current labels and, for KernelBench, the stable_above_eager boolean computed by the implemented tau=0.98 path. No threshold-robustness claim is made.

Future validation should preserve session_speedups for every candidate and rerun this script to compute threshold sensitivity directly.
