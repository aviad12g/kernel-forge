# Hardware And Software Environment

The final KernelBench pilot and repair artifacts preserve `environment_probe.json`. The final fused8 run directories are summarized in the local paper package, but their full environment probe files are not present in this workspace.

## Recorded KernelBench Environment

From `runs/20260520_202314/environment_probe.json` and `runs/20260520_213128/environment_probe.json`:

| Field | Value |
| --- | --- |
| Python | 3.12.3 |
| Platform | Linux 6.8.0-87-generic, x86_64 |
| Torch | 2.8.0+cu128 |
| CUDA available | yes |
| GPU | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 |
| Triton | 3.4.0 |
| Tiny Triton kernel | passed |
| Viability | TRITON_EXECUTION_OK |

The RunPod sprint notes report the same final GPU stack for the rigorous fused8 campaign: RTX 5090, Torch 2.8.0+cu128, Triton 3.4.0, and `TRITON_EXECUTION_OK`. The exact fused8 `environment_probe.json` files are not imported in this local workspace, so the paper should avoid implying that all low-level environment fields are locally preserved for fused8.

## Not Recorded

The current artifact set does not record:

- NVIDIA driver version;
- CUDA runtime driver/build version beyond the Torch `+cu128` build tag;
- locked GPU clocks;
- locked memory clocks;
- power limit;
- persistence mode;
- ECC status;
- thermals or ambient temperature;
- pod-level scheduling/noisy-neighbor conditions.

The paper limitations should state that clocks were not recorded as locked and that dynamic boost or thermal behavior may contribute to run-to-run variation.

## RunPod Flash Diagnostic Environment

Profiler diagnostics and headline clock-recorded validation were run later on a
RunPod Flash worker for preserved KernelBench loss candidates only. These
diagnostics do not replace the original benchmark environment or original
tables.

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 570.195.03 |
| Torch | 2.9.1+cu128 |
| Triton | 3.5.1 |
| Tiny Triton kernel | passed |
| Validation mode | clock-recorded |
| Clock locking | attempted, rejected by worker permissions |

The clock-recorded validation records graphics clock, memory clock, power, and
temperature before and after each preserved KernelBench loss candidate. Exact
fused8 headline candidate artifacts are not preserved locally, so fused8 rows
were marked unavailable rather than reconstructed.
