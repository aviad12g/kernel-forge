from openkernelforge import cli


class _OKBackend:
    def generate(self, prompt, *, system=None, **kwargs):
        return "OK"


class _FailingBackend:
    def generate(self, prompt, *, system=None, **kwargs):
        raise RuntimeError("backend unavailable")


def test_check_backend_success_with_mocked_backend(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("tasks: [vector_add]\nagent:\n  type: llm\n  backend: fake\n", encoding="utf-8")
    monkeypatch.setattr(cli, "create_backend", lambda agent_config: _OKBackend())
    code = cli.main(["check-backend", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Backend check succeeded" in captured.out


def test_check_backend_failure_with_mocked_backend(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("tasks: [vector_add]\nagent:\n  type: llm\n  backend: fake\n", encoding="utf-8")
    monkeypatch.setattr(cli, "create_backend", lambda agent_config: _FailingBackend())
    code = cli.main(["check-backend", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Backend check failed" in captured.out


def test_check_backend_does_not_call_disabled_config(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "tasks: [vector_add]\n"
        "agent:\n  type: llm\n  backend: fake\n"
        "execution:\n  disabled_reason: historical provenance only\n",
        encoding="utf-8",
    )

    def unexpected_backend(_agent_config):
        raise AssertionError("backend must not be created for a disabled config")

    monkeypatch.setattr(cli, "create_backend", unexpected_backend)
    code = cli.main(["check-backend", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Backend check blocked" in captured.out
