# KernelBench Adapter Contract

## Official Model Contract

KernelBench task files define `Model`, `get_init_inputs()`, and `get_inputs()`.
Generated implementations for these tasks use the official `ModelNew`
contract: `ModelNew` receives the same initialization arguments as `Model`, and
`ModelNew.forward` receives the values returned by `get_inputs()`.

OpenKernelForge materializes `get_init_inputs()` once under fixed seed `0` and
clones that snapshot into the reference `Model` and candidate `ModelNew`.
Both modules are then moved to the target dtype/device before verification or
timing and reused. Every official KernelBench task that defines `Model`
requires `ModelNew`, including tasks whose `state_dict()` is empty. A free
module-level `forward(*args)` remains available only for local synthetic tasks
that define the separate `reference_fn`/`input_generator` interface.

Correctness verification clones the same generated inputs for reference and
candidate execution. It compares outputs and final input state. A candidate
that mutates an input when the reference does not is rejected; when the
reference is in-place, the candidate must produce the same input side effects.

Each run record now includes:

- `candidate_contract`;
- `reference_has_model_state`;
- `reference_state_keys` in task metadata;
- `model_init_seed`;
- static-policy version.

## Timing Lifecycle

Reference and candidate modules are prepared before warmup and measured
samples. The timed callable is the persistent module's forward execution, not
module construction, parameter initialization, or `.to(device)` transfer.
The compiled baseline wraps the same persistent reference callable. The first
materializing compiled invocation is accounted for separately from steady-state
runtime.

For official tasks, shape discovery and the memory-cap preflight first invoke
`get_inputs()` under PyTorch's `meta` device. This records tensor shape, dtype,
and byte count without allocating the full payload. If an official task cannot
run on `meta`, the default is to skip it with
`MEMORY_ESTIMATE_UNAVAILABLE`; real CPU materialization requires the explicit
`allow_cpu_memory_preflight_fallback` config flag. The
memory estimate accounts for the five input-tree copies held by correctness
verification, two copies of known module state when state exists, and configured
cache-perturbation storage. It remains a lower bound because outputs, temporary
workspace, compiler workspace, and allocator fragmentation are unknown.

## Historical Pilot Status

Runs `20260520_202314` and `20260520_213128` predate this corrected contract.
Their adapter required a free `forward(*args)` candidate, did not preserve the
official `ModelNew` lifecycle, and reconstructed the reference `Model` inside
each call. The mismatch is structurally invalid for every official `Model`
task and makes missing state explicit on parameterized tasks such as
convolution. The preserved
candidate, verification, timing, taxonomy, and repair artifacts remain useful
for auditing how the defect was discovered, but their aggregate correctness
rate and speedups are provisional until rerun through the corrected adapter.

Historical files are not modified or silently relabeled.

## Security Boundary

Static policy checks run before local import. Local import is still in-process
and is not an operating-system sandbox. Untrusted public submissions should be
evaluated in a disposable worker or container with an external timeout and no
secrets in the environment.
