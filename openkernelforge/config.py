"""Configuration helpers for OpenKernelForge runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ALLOWED_TOP_LEVEL_KEYS = {
    "tasks",
    "output_dir",
    "agent",
    "verification",
    "benchmark",
    "execution",
    "kernelbench",
}


@dataclass
class VerificationConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    dtype: str = "float32"
    device: str = "auto"
    max_shapes_per_task: int = 1

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("verification.seeds must not be empty")
        if int(self.max_shapes_per_task) <= 0:
            raise ValueError("verification.max_shapes_per_task must be positive")


@dataclass
class CacheFlushBenchmarkConfig:
    enabled: bool = False
    size_mb: int = 128
    mode: str = "write"

    def __post_init__(self) -> None:
        if int(self.size_mb) <= 0:
            raise ValueError("benchmark.cache_flush.size_mb must be positive")
        if self.mode not in {"write", "read_write"}:
            raise ValueError("benchmark.cache_flush.mode must be 'write' or 'read_write'")


@dataclass
class BootstrapCIConfig:
    enabled: bool = False
    samples: int = 1000
    seed: int = 123

    def __post_init__(self) -> None:
        if int(self.samples) <= 0:
            raise ValueError("benchmark.bootstrap_ci.samples must be positive")


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    timing_mode: str = "auto"
    warmup: int = 5
    repeats: int = 20
    independent_sessions: int = 1
    dtype: str = "float32"
    device: str = "auto"
    max_shapes_per_task: int = 1
    enable_torch_compile: bool = False
    torch_compile_mode: str | None = None
    cache_flush: CacheFlushBenchmarkConfig = field(default_factory=CacheFlushBenchmarkConfig)
    bootstrap_ci: BootstrapCIConfig = field(default_factory=BootstrapCIConfig)
    separate_compile_time: bool = True
    stable_session_threshold: float = 0.98

    def __post_init__(self) -> None:
        if isinstance(self.cache_flush, dict):
            self.cache_flush = CacheFlushBenchmarkConfig(**self.cache_flush)
        if isinstance(self.bootstrap_ci, dict):
            self.bootstrap_ci = BootstrapCIConfig(**self.bootstrap_ci)
        if self.timing_mode not in {"auto", "wall_clock", "cuda_event"}:
            raise ValueError("benchmark.timing_mode must be auto, wall_clock, or cuda_event")
        if int(self.warmup) < 0:
            raise ValueError("benchmark.warmup must be non-negative")
        if int(self.repeats) <= 0:
            raise ValueError("benchmark.repeats must be positive")
        if int(self.independent_sessions) <= 0:
            raise ValueError("benchmark.independent_sessions must be positive")
        if int(self.max_shapes_per_task) <= 0:
            raise ValueError("benchmark.max_shapes_per_task must be positive")
        if not 0.0 < float(self.stable_session_threshold) <= 1.0:
            raise ValueError("benchmark.stable_session_threshold must be in (0, 1]")


@dataclass
class PerformanceSearchConfig:
    enabled: bool = False
    mode: str = "performance_search"
    max_rounds: int = 3
    candidates_per_round: int = 4
    target_speedup_vs_eager: float | None = 1.0
    target_speedup_vs_compile: float | None = None
    keep_top_k: int = 3
    optimize_only_selected_best: bool = True
    include_best_template_context: bool = False
    template_run_dir: str | None = None
    parameter_grid: dict[str, Any] = field(default_factory=dict)
    candidates_per_setting: int = 1
    max_settings_per_task: int | None = None


@dataclass
class TemplateCopyConfig:
    reject_if_preservation_score_below: int = 70
    reject_fallbacks: bool = True
    reject_forbidden_torch_ops: bool = True


@dataclass
class AgentConfig:
    type: str = "dummy"
    backend: str = "fake"
    provider: str | None = None
    api_mode: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    max_attempts: int = 3
    candidates_per_attempt: int = 1
    stop_after_first_correct: bool = True
    benchmark_all_correct: bool = True
    allow_torch_fallback: bool = True
    temperature: float = 0.2
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    fake_mode: str = "correct"
    backend_options: dict[str, Any] = field(default_factory=dict)
    template_family: str = "elementwise"
    template_variants: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = "v1_default"
    repair_prompt_version: str = "v1_default"
    performance_prompt_version: str = "v1_cuda_elementwise_perf"
    performance_search: PerformanceSearchConfig = field(default_factory=PerformanceSearchConfig)
    template_copy: TemplateCopyConfig = field(default_factory=TemplateCopyConfig)

    def __post_init__(self) -> None:
        if isinstance(self.performance_search, dict):
            self.performance_search = PerformanceSearchConfig(**self.performance_search)
        if isinstance(self.template_copy, dict):
            self.template_copy = TemplateCopyConfig(**self.template_copy)
        if int(self.max_attempts) <= 0:
            raise ValueError("agent.max_attempts must be positive")
        if int(self.candidates_per_attempt) <= 0:
            raise ValueError("agent.candidates_per_attempt must be positive")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("agent.timeout_seconds must be positive")
        if self.temperature is not None and float(self.temperature) < 0:
            raise ValueError("agent.temperature must be non-negative")
        if self.top_p is not None and not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("agent.top_p must be in (0, 1]")


@dataclass
class ExecutionConfig:
    require_cuda: bool = False
    require_triton: bool = False
    require_tiny_triton_kernel: bool = False
    disabled_reason: str | None = None


@dataclass
class RunConfig:
    tasks: list[str] = field(default_factory=lambda: ["vector_add", "relu", "row_sum"])
    output_dir: str = "runs"
    agent: AgentConfig = field(default_factory=AgentConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        unknown = sorted(set(data).difference(_ALLOWED_TOP_LEVEL_KEYS))
        if unknown:
            raise ValueError("Unknown top-level config fields: " + ", ".join(unknown))
        tasks_data = data.get("tasks", cls().tasks)
        if isinstance(tasks_data, dict):
            tasks = list(tasks_data.get("include", []))
        else:
            tasks = list(tasks_data)

        verification = VerificationConfig(**data.get("verification", {}))
        benchmark_data = dict(data.get("benchmark", {}))
        if "repeat" in benchmark_data and "repeats" not in benchmark_data:
            benchmark_data["repeats"] = benchmark_data.pop("repeat")
        if "include_torch_compile" in benchmark_data and "enable_torch_compile" not in benchmark_data:
            benchmark_data["enable_torch_compile"] = benchmark_data.pop("include_torch_compile")
        benchmark = BenchmarkConfig(**benchmark_data)
        agent = AgentConfig(**data.get("agent", {}))
        execution = ExecutionConfig(**data.get("execution", {}))
        return cls(
            tasks=tasks,
            output_dir=str(data.get("output_dir", "runs")),
            agent=agent,
            verification=verification,
            benchmark=benchmark,
            execution=execution,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_safe_dict(self) -> dict[str, Any]:
        return redact_secrets(self.to_dict())


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return RunConfig.from_dict(data)


def save_config(config: RunConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_safe_dict(), f, sort_keys=False)


def redact_secrets(value: Any) -> Any:
    """Return a copy with API keys and auth-like fields redacted."""

    secret_keys = {
        "api_key",
        "authorization",
        "access_token",
        "refresh_token",
        "bearer_token",
        "secret",
        "password",
    }
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if (
                lowered in secret_keys
                or lowered.endswith("_api_key")
                or lowered.endswith("_secret")
                or lowered.endswith("_password")
            ):
                redacted[key] = "<redacted>" if item else item
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value
