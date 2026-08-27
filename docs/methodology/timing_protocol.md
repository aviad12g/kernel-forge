# Timing Protocol

The rigorous timing path is implemented in `openkernelforge/harness/timing.py`, `openkernelforge/harness/cache_flush.py`, and `openkernelforge/harness/benchmarker.py`.

## Timing Primitive

CUDA runs use `CudaEventTimer`, which creates `torch.cuda.Event(enable_timing=True)` start/end events for each measured sample. The timer:

1. runs the configured warmup iterations;
2. synchronizes the CUDA device;
3. optionally perturbs cache state before each measured sample;
4. records the start event;
5. runs the function;
6. records the end event;
7. synchronizes the CUDA device;
8. records `start.elapsed_time(end)` in milliseconds.

CPU-only tests use `WallClockTimer`, a development fallback based on `time.perf_counter` with explicit device synchronization where applicable. Paper-facing GPU results use CUDA events.

## Warmup, Samples, And Sessions

For the rigorous fused8 runs, the configured protocol is:

- warmup: 30 iterations;
- measured samples: 120 per session;
- same-process measurement sessions: 3;
- timing mode: `cuda_event`;
- compiler baseline: `torch.compile` with mode `max-autotune`;
- compile time separated from runtime where available.

For the KernelBench artifacts, the safe baseline validation used 100 measured samples per session. The Gemini candidate and repair configs use 120 measured samples per session. The paper table therefore reports `100/120` rather than a single number.

A measurement session is a separate benchmark loop inside the same process and same GPU run. For session index `i`, inputs are regenerated with seed `1234 + i`, and a fresh timer/cache-flusher object is used. Sessions are not separate OS processes and do not imply GPU clock reset. Current runs rotate eager, candidate, and compiled measurement order between sessions to reduce systematic order effects; imported historical artifacts used fixed order.

## Cache-State Perturbation

Cache flushing is best described as cache-state perturbation, not a guaranteed complete L2 flush.

The current `CudaCacheFlusher`:

- allocates one CUDA `float32` buffer per benchmark session on first use;
- default buffer size is 128 MB;
- mode `write`: fills the buffer with `1.0` before each measured sample;
- mode `read_write`: computes `buffer.sum()` and then fills the buffer;
- does nothing on CPU or when CUDA is unavailable, recording a warning instead.

The rigorous fused8 and KernelBench model/candidate configs enable `cache_flush.enabled = true`, `size_mb = 128`, and `mode = write`. The flush is called after warmup and before each measured sample for eager, candidate, and compiled measurements. It is intended to reduce dependence on a favorable residual cache state, not to guarantee full eviction of every relevant cache level on RTX 5090 class hardware.

## Summaries

The sample summarizer reports:

- `n`;
- mean;
- median;
- p25 and p75;
- IQR;
- min and max;
- standard deviation;
- coefficient of variation;
- deterministic bootstrap median CI when enabled.

Bootstrap uses a local deterministic RNG seed, default `123`, and default `1000` resamples. Timing-level intervals describe within-session sample variation. The current benchmarker additionally summarizes the per-session speedup vector and bootstraps that vector when enabled; with only three sessions, this interval is descriptive rather than a high-powered estimate of between-run variance. Historical pooled timing intervals must not be interpreted as independent-session uncertainty.

## Compile-Time Accounting

PyTorch compilation is lazy. The current benchmarker records
`compile_time_kind = wrapper_and_first_call` and measures the wall-clock time
from `torch.compile(...)` through the first synchronized materializing call.
Steady-state runtime samples occur afterward and exclude that first call.
Historical `compile_time_ms` fields that measured wrapper construction alone
are retained as historical artifacts but are not interpreted as compilation
cost.

## Stability Fields

The benchmarker records per-session speedups and sets:

```text
stable_above_eager =
    median(session_speedups) >= 1.0
    and all(session_speedup >= stable_session_threshold for session_speedup in session_speedups)
```

The default `stable_session_threshold` is `0.98`.
