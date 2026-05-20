"""Configuration helpers for OpenKernelForge runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VerificationConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    dtype: str = "float32"
    device: str = "auto"
    max_shapes_per_task: int = 1


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    warmup: int = 5
    repeats: int = 20
    dtype: str = "float32"
    device: str = "auto"
    max_shapes_per_task: int = 1
    enable_torch_compile: bool = False


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


@dataclass
class ExecutionConfig:
    require_cuda: bool = False
    require_triton: bool = False
    require_tiny_triton_kernel: bool = False


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
