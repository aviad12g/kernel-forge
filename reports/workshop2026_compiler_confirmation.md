# Fresh-Process Compiler-Rung Confirmation

One frozen generated candidate was re-evaluated against `torch.compile
max-autotune` in seven fresh RTX A4500 processes. This is a separate
compiler-rung control, not a replacement for the primary candidate-versus-eager
campaign.

- Task: `Matmul_with_diagonal_matrices`.
- Candidate: `candidate_000`, SHA-256
  `8d0461a11b117d7e188ff989f0cdae90ea840f770a4c333e5d09a82027dbec6e`.
- Median candidate speedup versus compile: `2.001165x`.
- Process-median range: `1.993307x` to `2.003077x`.
- Fresh processes: 7/7 complete.
- Runtime comparison excludes compilation; compile-and-first-call latency is
  preserved in the raw records.

The candidate remains below eager in the original primary screening result
(`0.937x`). The supported conclusion is therefore specific: this candidate
beats the compiler baseline but not the eager library path.
