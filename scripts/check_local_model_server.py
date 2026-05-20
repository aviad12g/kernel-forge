from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.config import load_config


@dataclass
class LocalServerCheckResult:
    available: bool
    base_url: str
    model: str | None = None
    models: list[str] = field(default_factory=list)
    generation_ok: bool = False
    message: str = ""


def check_local_model_server(
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 10.0,
) -> LocalServerCheckResult:
    """Check an OpenAI-compatible local server without requiring an API key."""

    normalized_base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    models: list[str] = []
    models_error = ""
    try:
        models_data = _request_json(
            "GET",
            f"{normalized_base}/models",
            headers=headers,
            timeout=timeout,
        )
        raw_models = models_data.get("data", []) if isinstance(models_data, dict) else []
        models = [
            str(item.get("id"))
            for item in raw_models
            if isinstance(item, dict) and item.get("id")
        ]
    except Exception as exc:  # keep checker defensive across local server variants
        models_error = str(exc)

    selected_model = model or (models[0] if models else None)
    if not selected_model:
        return LocalServerCheckResult(
            available=False,
            base_url=normalized_base,
            models=models,
            message=(
                "Local server did not expose a model list and no --model/config model was provided. "
                f"/models error: {models_error or 'none'}"
            ),
        )

    body = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "Return only OK."},
            {"role": "user", "content": "Return the word OK."},
        ],
        "max_tokens": 8,
    }
    try:
        completion = _request_json(
            "POST",
            f"{normalized_base}/chat/completions",
            payload=body,
            headers=headers,
            timeout=timeout,
        )
        content = _extract_chat_content(completion)
        if "OK" not in content.upper():
            return LocalServerCheckResult(
                available=False,
                base_url=normalized_base,
                model=selected_model,
                models=models,
                message="Chat completion succeeded but response did not contain OK.",
            )
    except Exception as exc:
        return LocalServerCheckResult(
            available=False,
            base_url=normalized_base,
            model=selected_model,
            models=models,
            message=f"Chat completion health check failed: {exc}",
        )

    return LocalServerCheckResult(
        available=True,
        base_url=normalized_base,
        model=selected_model,
        models=models,
        generation_ok=True,
        message="Local OpenAI-compatible server is available.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a local OpenAI-compatible model server.")
    parser.add_argument("--config", help="Optional OpenKernelForge config to read base_url/model from")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    base_url = args.base_url
    model = args.model
    api_key_env = args.api_key_env
    if args.config:
        config = load_config(args.config)
        base_url = config.agent.base_url or base_url
        model = model or config.agent.model
        api_key_env = api_key_env if api_key_env is not None else config.agent.api_key_env

    result = check_local_model_server(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout=args.timeout,
    )
    print(f"Local server: {result.base_url}")
    if result.models:
        print("Available models:")
        for name in result.models:
            print(f"- {name}")
    elif result.model:
        print(f"Configured model: {result.model}")
    print(result.message)
    return 0 if result.available else 1


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {_truncate(body)}") from exc
    except error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON response: {_truncate(raw)}") from exc


def _extract_chat_content(data: Any) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed chat completion response: {_truncate(json.dumps(data))}") from exc


def _truncate(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
