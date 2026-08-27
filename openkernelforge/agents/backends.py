"""Provider-agnostic model backend interfaces and test backends."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol

from openkernelforge.config import AgentConfig
from openkernelforge.agents.dummy_agent import torch_fallback_candidate


class ModelBackend(Protocol):
    """Minimal interface any text-generation backend must implement."""

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """Return a raw model response for a prompt."""


@dataclass
class FakeBackend:
    """Deterministic backend for tests and offline smoke runs.

    Modes:
    - ``correct``: always returns a correct torch fallback.
    - ``broken_then_fixed``: first response per task is wrong, later responses are correct.
    - ``always_broken``: always returns an incorrect candidate.
    """

    mode: str = "correct"
    fenced: bool = True
    calls: list[dict[str, object]] = field(default_factory=list)
    _task_counts: dict[str, int] = field(default_factory=dict)
    last_response_metadata: dict[str, object] = field(default_factory=dict, init=False)

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        self.last_response_metadata = {
            "backend": "fake",
            "model": "fake-deterministic",
            "provider_response_model": "fake-deterministic",
        }
        if "Return the word OK" in prompt:
            self.calls.append(
                {
                    "task_id": "health_check",
                    "system": system,
                    "prompt": prompt,
                    "kwargs": dict(kwargs),
                }
            )
            return "OK"

        task_id = _extract_task_id(prompt)
        count = self._task_counts.get(task_id, 0)
        self._task_counts[task_id] = count + 1
        self.calls.append(
            {
                "task_id": task_id,
                "system": system,
                "prompt": prompt,
                "kwargs": dict(kwargs),
            }
        )

        if self.mode not in {"correct", "broken_then_fixed", "always_broken"}:
            raise ValueError(f"Unknown FakeBackend mode: {self.mode}")

        if self.mode == "always_broken" or (self.mode == "broken_then_fixed" and count == 0):
            code = _broken_candidate(task_id)
        else:
            code = torch_fallback_candidate(task_id)

        if self.fenced:
            return f"```python\n{code}\n```"
        return code


@dataclass
class LocalCommandBackend:
    """Backend that delegates generation to a local command over stdin/stdout."""

    command: list[str]
    timeout_seconds: float = 120.0

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        if not self.command:
            raise RuntimeError("LocalCommandBackend requires a non-empty command")
        payload = json.dumps({"prompt": prompt, "system": system, "kwargs": kwargs})
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Local command backend timed out after {self.timeout_seconds} seconds"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "Local command backend failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        return completed.stdout


@dataclass
class OpenAICompatibleBackend:
    """Client for OpenAI-compatible ``/v1/chat/completions`` servers."""

    base_url: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, object] = field(default_factory=dict)
    last_response_metadata: dict[str, object] = field(default_factory=dict, init=False)

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        requests = _load_requests()

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            **self.extra_body,
        }
        _set_if_not_none(body, "temperature", kwargs.get("temperature", self.temperature))
        _set_if_not_none(body, "top_p", kwargs.get("top_p", self.top_p))
        _set_if_not_none(body, "max_tokens", kwargs.get("max_tokens", self.max_tokens))

        headers = {"Content-Type": "application/json", **self.extra_headers}
        api_key = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        try:
            response = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OpenAI-compatible backend timed out after {self.timeout_seconds} seconds"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"OpenAI-compatible backend connection failed: {exc}") from exc

        if response.status_code != 200:
            detail = _truncate(response.text.strip(), 1000)
            raise RuntimeError(
                f"OpenAI-compatible backend returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenAI-compatible backend returned malformed JSON") from exc

        content = _extract_chat_content(data)
        if not content.strip():
            raise RuntimeError("OpenAI-compatible backend returned empty assistant content")
        self.last_response_metadata = _response_metadata(
            data,
            configured_model=self.model,
            backend="openai_compatible",
            endpoint=url,
            response=response,
        )
        return content


@dataclass
class OpenAIResponsesBackend:
    """Minimal client for OpenAI's native ``/v1/responses`` API."""

    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str | None = "OPENAI_API_KEY"
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, object] = field(default_factory=dict)
    last_response_metadata: dict[str, object] = field(default_factory=dict, init=False)

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        requests = _load_requests()
        if not self.model:
            raise RuntimeError("OpenAIResponsesBackend requires model")
        api_key = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if not api_key:
            env = self.api_key_env or "OPENAI_API_KEY"
            raise RuntimeError(f"OpenAI Responses backend requires API key: export {env}=<your-key>")

        input_text = prompt if not system else f"{system}\n\n{prompt}"
        body = {
            "model": self.model,
            "input": input_text,
            **self.extra_body,
        }
        _set_if_not_none(body, "temperature", kwargs.get("temperature", self.temperature))
        _set_if_not_none(body, "top_p", kwargs.get("top_p", self.top_p))
        max_output_tokens = kwargs.get("max_output_tokens", kwargs.get("max_tokens", self.max_tokens))
        _set_if_not_none(body, "max_output_tokens", max_output_tokens)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **self.extra_headers,
        }
        url = self.base_url.rstrip("/") + "/responses"
        try:
            response = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OpenAI Responses backend timed out after {self.timeout_seconds} seconds"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"OpenAI Responses backend connection failed: {exc}") from exc

        if response.status_code != 200:
            detail = _truncate(response.text.strip(), 1000)
            raise RuntimeError(f"OpenAI Responses backend returned HTTP {response.status_code}: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenAI Responses backend returned malformed JSON") from exc
        content = _extract_responses_content(data)
        if not content.strip():
            raise RuntimeError("OpenAI Responses backend returned empty output text")
        self.last_response_metadata = _response_metadata(
            data,
            configured_model=self.model,
            backend="openai_responses",
            endpoint=url,
            response=response,
        )
        return content


@dataclass
class TransformersBackend:
    """Placeholder for future in-process Transformers generation."""

    model_name_or_path: str

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        raise RuntimeError(
            "TransformersBackend is a placeholder in this phase. "
            "Use backend=fake or add a real local model integration in the next phase."
        )


def create_backend(
    agent_config: AgentConfig | str,
    *,
    options: dict[str, object] | None = None,
) -> ModelBackend:
    """Create a model backend from ``AgentConfig``.

    A string name plus ``options`` remains supported for older tests and callers.
    """

    if isinstance(agent_config, AgentConfig):
        name = agent_config.backend
        opts = _backend_options_from_agent_config(agent_config)
    else:
        name = agent_config
        opts = dict(options or {})

    normalized = name.replace("_", "-").lower()
    if normalized == "fake":
        return FakeBackend(
            mode=str(opts.get("fake_mode", opts.get("mode", "correct"))),
            fenced=bool(opts.get("fenced", True)),
        )
    if normalized in {"local-command", "command"}:
        command = opts.get("command")
        if isinstance(command, str):
            command_list = command.split()
        elif isinstance(command, list):
            command_list = [str(part) for part in command]
        else:
            raise RuntimeError("LocalCommandBackend requires backend_options.command")
        return LocalCommandBackend(
            command=command_list,
            timeout_seconds=_float_option(opts, "timeout_seconds", 120.0),
        )
    if normalized in {"openai-responses", "responses"}:
        model = opts.get("model")
        if not model:
            raise RuntimeError("OpenAIResponsesBackend requires model")
        return OpenAIResponsesBackend(
            base_url=str(opts.get("base_url") or "https://api.openai.com/v1"),
            model=str(model),
            api_key_env=_optional_str(opts.get("api_key_env")) or "OPENAI_API_KEY",
            api_key=_optional_str(opts.get("api_key")),
            temperature=_optional_float(opts.get("temperature")),
            top_p=_optional_float(opts.get("top_p")),
            max_tokens=_optional_int(opts.get("max_tokens")),
            timeout_seconds=_float_option(opts, "timeout_seconds", 120.0),
            extra_headers=_str_dict(opts.get("extra_headers")),
            extra_body=_object_dict(opts.get("extra_body"), field_name="extra_body"),
        )
    if normalized in {"openai-compatible", "openai"}:
        base_url = opts.get("base_url")
        model = opts.get("model")
        if not base_url or not model:
            raise RuntimeError(
                "OpenAICompatibleBackend requires base_url and model"
            )
        return OpenAICompatibleBackend(
            base_url=str(base_url),
            model=str(model),
            api_key_env=_optional_str(opts.get("api_key_env")),
            api_key=_optional_str(opts.get("api_key")),
            temperature=_optional_float(opts.get("temperature")),
            top_p=_optional_float(opts.get("top_p")),
            max_tokens=_optional_int(opts.get("max_tokens")),
            timeout_seconds=_float_option(opts, "timeout_seconds", 120.0),
            extra_headers=_str_dict(opts.get("extra_headers")),
            extra_body=_object_dict(opts.get("extra_body"), field_name="extra_body"),
        )
    if normalized == "transformers":
        model_name_or_path = opts.get("model_name_or_path")
        if not model_name_or_path:
            raise RuntimeError("TransformersBackend requires backend_options.model_name_or_path")
        return TransformersBackend(model_name_or_path=str(model_name_or_path))
    raise ValueError(f"Unknown backend '{name}'")


def _backend_options_from_agent_config(agent_config: AgentConfig) -> dict[str, object]:
    opts = dict(agent_config.backend_options or {})
    for key in (
        "model",
        "provider",
        "api_mode",
        "base_url",
        "api_key_env",
        "api_key",
        "temperature",
        "top_p",
        "max_tokens",
        "timeout_seconds",
        "extra_headers",
        "extra_body",
        "fake_mode",
    ):
        value = getattr(agent_config, key)
        if value is not None and value != {}:
            opts.setdefault(key, value)
    return opts


def _load_requests() -> ModuleType:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "OpenAICompatibleBackend requires the optional 'requests' package. "
            "Install it with `pip install requests`."
        ) from exc
    return requests


def _extract_chat_content(data: object) -> str:
    try:
        if not isinstance(data, dict):
            raise TypeError("response JSON is not an object")
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise TypeError("choices is missing or empty")
        first = choices[0]
        if not isinstance(first, dict):
            raise TypeError("first choice is not an object")
        message = first["message"]
        if not isinstance(message, dict):
            raise TypeError("message is not an object")
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("message.content is not a string")
        return content
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError(f"Malformed OpenAI-compatible response: {data}") from exc


def _extract_responses_content(data: object) -> str:
    if not isinstance(data, dict):
        raise RuntimeError(f"Malformed OpenAI Responses response: {data}")
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError(f"Malformed OpenAI Responses response: {data}")


def _set_if_not_none(body: dict[str, object], key: str, value: object) -> None:
    if value is not None:
        body[key] = value


def _response_metadata(
    data: object,
    *,
    configured_model: str,
    backend: str,
    endpoint: str,
    response: object,
) -> dict[str, object]:
    payload = data if isinstance(data, dict) else {}
    metadata: dict[str, object] = {
        "backend": backend,
        "configured_model": configured_model,
        "provider_response_model": payload.get("model") or "not_returned",
        "response_id": payload.get("id"),
        "created": payload.get("created"),
        "object": payload.get("object"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "usage": payload.get("usage"),
        "endpoint": endpoint,
    }
    headers = getattr(response, "headers", None)
    if headers is not None:
        request_id = headers.get("x-request-id") or headers.get("request-id")
        if request_id:
            metadata["request_id"] = str(request_id)
    return metadata


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Expected a numeric value, got {type(value).__name__}")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (str, int)):
        raise TypeError(f"Expected an integer value, got {type(value).__name__}")
    return int(value)


def _str_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("extra_headers must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _object_dict(value: object, *, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _float_option(options: dict[str, object], key: str, default: float) -> float:
    value = _optional_float(options.get(key))
    return default if value is None else value


def _extract_task_id(prompt: str) -> str:
    match = re.search(r"Task id:\s*([A-Za-z0-9_./-]+)", prompt)
    return match.group(1) if match else "unknown"


def _broken_candidate(task_id: str) -> str:
    if task_id in {"vector_add", "elementwise_mul", "sigmoid_mul"}:
        return "import torch\n\n\ndef forward(x, y):\n    return x - y"
    if task_id == "relu":
        return "import torch\n\n\ndef forward(x):\n    return -x"
    if task_id == "bias_relu":
        return "import torch\n\n\ndef forward(x, bias):\n    return x + bias"
    if task_id == "row_sum":
        return "import torch\n\n\ndef forward(x):\n    return torch.sum(x, dim=0)"
    if task_id == "layernorm_small":
        return "import torch\n\n\ndef forward(x, weight, bias):\n    return x"
    if task_id == "matmul_bias":
        return "import torch\n\n\ndef forward(x, weight, bias):\n    return x @ weight"
    return "import torch\n\n\ndef forward(*args):\n    return args[0]"
