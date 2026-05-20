"""Optional CUDA cache flushing for benchmark measurements."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheFlushConfig:
    enabled: bool = False
    size_mb: int = 128
    mode: str = "write"

    def __post_init__(self) -> None:
        self.size_mb = int(self.size_mb)
        if self.mode not in {"write", "read_write"}:
            raise ValueError("cache flush mode must be 'write' or 'read_write'")


class CudaCacheFlusher:
    """Best-effort CUDA cache flusher used between measured samples."""

    def __init__(self, config: CacheFlushConfig | dict | None, *, device: torch.device) -> None:
        if isinstance(config, dict):
            config = CacheFlushConfig(**config)
        self.config = config or CacheFlushConfig()
        self.device = torch.device(device)
        self.cache_flush_enabled = bool(self.config.enabled)
        self.cache_flush_performed = False
        self.warning: str | None = None
        self._buffer: torch.Tensor | None = None

    def flush(self) -> bool:
        """Flush by touching a large CUDA buffer; return whether work happened."""

        if not self.config.enabled:
            return False
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.warning = "cache flush requested but CUDA is unavailable"
            return False

        try:
            if self._buffer is None:
                numel = max(1, (self.config.size_mb * 1024 * 1024) // 4)
                self._buffer = torch.empty(numel, device=self.device, dtype=torch.float32)
            if self.config.mode == "read_write":
                _ = self._buffer.sum()
            self._buffer.fill_(1.0)
            self.cache_flush_performed = True
            return True
        except Exception as exc:  # pragma: no cover - hardware-dependent defensive path
            self.warning = f"cache flush failed: {exc}"
            return False
