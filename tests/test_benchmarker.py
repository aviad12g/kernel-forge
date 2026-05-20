from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.tasks.simple_tasks import get_task


def test_benchmarker_returns_positive_runtime():
    task = get_task("vector_add")

    def forward(x, y):
        return x + y

    result = benchmark_task(
        task,
        forward,
        shape=(16,),
        device="cpu",
        dtype="float32",
        warmup=1,
        repeats=3,
        enable_torch_compile=False,
    )
    assert result.benchmark_error is None
    assert result.eager is not None
    assert result.candidate is not None
    assert result.eager.median_ms > 0
    assert result.candidate.median_ms > 0
