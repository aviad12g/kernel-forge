import pytest

import openkernelforge.agents.backends as backends_module
from openkernelforge.agents.backends import (
    FakeBackend,
    LocalCommandBackend,
    OpenAICompatibleBackend,
    create_backend,
)
from openkernelforge.config import AgentConfig


def test_fake_backend_returns_deterministic_correct_output():
    prompt = "Task id: vector_add\n"
    backend = FakeBackend(mode="correct", fenced=False)
    first = backend.generate(prompt)
    second = backend.generate(prompt)
    assert first == second
    assert "return x + y" in first
    assert len(backend.calls) == 2


def test_fake_backend_can_return_broken_then_fixed_output():
    prompt = "Task id: vector_add\n"
    backend = FakeBackend(mode="broken_then_fixed", fenced=False)
    first = backend.generate(prompt)
    second = backend.generate(prompt)
    assert "return x - y" in first
    assert "return x + y" in second


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeRequests:
    class exceptions:
        class Timeout(Exception):
            pass

        class RequestException(Exception):
            pass

    def __init__(self, post):
        self.post = post


def test_openai_compatible_backend_parses_valid_response(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(
            payload={
                "id": "response-1",
                "model": "provider-model-version",
                "usage": {"total_tokens": 12},
                "choices": [{"message": {"content": "def forward(x):\n    return x\n"}}],
            },
            headers={"x-request-id": "request-1"},
        )

    monkeypatch.setattr(backends_module, "_load_requests", lambda: _FakeRequests(fake_post))
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model="local-model",
        api_key="secret-value",
        temperature=0.2,
        top_p=0.95,
        max_tokens=128,
        timeout_seconds=3,
    )
    content = backend.generate("Task id: relu", system="system")
    assert "def forward" in content
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["model"] == "local-model"
    assert captured["json"]["temperature"] == 0.2
    assert captured["json"]["top_p"] == 0.95
    assert captured["json"]["max_tokens"] == 128
    assert captured["headers"]["Authorization"] == "Bearer secret-value"
    assert captured["timeout"] == 3
    assert backend.last_response_metadata["configured_model"] == "local-model"
    assert backend.last_response_metadata["provider_response_model"] == "provider-model-version"
    assert backend.last_response_metadata["request_id"] == "request-1"


def test_openai_compatible_backend_handles_non_200(monkeypatch):
    fake_requests = _FakeRequests(
        lambda *args, **kwargs: _FakeResponse(status_code=500, text="server failed")
    )
    monkeypatch.setattr(backends_module, "_load_requests", lambda: fake_requests)
    backend = OpenAICompatibleBackend(base_url="http://localhost:8000/v1", model="m")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        backend.generate("prompt")


def test_openai_compatible_backend_handles_empty_and_malformed_response(monkeypatch):
    backend = OpenAICompatibleBackend(base_url="http://localhost:8000/v1", model="m")

    empty_requests = _FakeRequests(
        lambda *args, **kwargs: _FakeResponse(payload={"choices": [{"message": {"content": ""}}]})
    )
    monkeypatch.setattr(backends_module, "_load_requests", lambda: empty_requests)
    with pytest.raises(RuntimeError, match="empty assistant content"):
        backend.generate("prompt")

    malformed_requests = _FakeRequests(lambda *args, **kwargs: _FakeResponse(payload={"unexpected": []}))
    monkeypatch.setattr(backends_module, "_load_requests", lambda: malformed_requests)
    with pytest.raises(RuntimeError, match="Malformed"):
        backend.generate("prompt")


def test_backend_factory_creates_fake_backend():
    backend = create_backend(AgentConfig(type="llm", backend="fake", fake_mode="correct"))
    assert isinstance(backend, FakeBackend)


def test_local_command_backend_enforces_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise backends_module.subprocess.TimeoutExpired(cmd=["backend"], timeout=3)

    monkeypatch.setattr(backends_module.subprocess, "run", timeout)
    backend = LocalCommandBackend(command=["backend"], timeout_seconds=3)

    with pytest.raises(RuntimeError, match="timed out after 3 seconds"):
        backend.generate("prompt")


def test_backend_factory_creates_openai_compatible_backend():
    backend = create_backend(
        AgentConfig(
            type="llm",
            backend="openai_compatible",
            model="local-model",
            base_url="http://localhost:8000/v1",
            api_key_env="OPENAI_API_KEY",
            temperature=0.1,
        )
    )
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model == "local-model"
    assert backend.base_url == "http://localhost:8000/v1"
