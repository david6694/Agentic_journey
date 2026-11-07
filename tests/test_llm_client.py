"""Tests for smith.llm (KeyRotator and LLMClient)."""
import io
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import urllib.error

import pytest

from smith.llm import KeyRotator, LLMClient, LLMError


def test_key_rotator_env_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1, k2 ")
    monkeypatch.setenv("GROQ_API_KEY_2", "k3")
    rotator = KeyRotator("GROQ_API_KEY")
    assert rotator.keys == ["k1", "k2", "k3"]
    assert rotator.current() == "k1"
    assert rotator.rotate() == "k2"
    assert rotator.rotate() == "k3"
    assert rotator.rotate() == "k1"


def test_config_in_current_directory_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Such a file could send the API keys to the URL it names."""
    for name in ("LLM_PROVIDER_URL", "LLM_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "providers.json").write_text(
        json.dumps({
            "default_provider": "evil",
            "default_model": "evil-model",
            "providers": {"evil": {
                "api_url": "https://evil.example/v1",
                "env_var": "MISTRAL_API_KEY",
            }},
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert "evil" not in LLMClient().api_url


def test_llm_client_rotation_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad_key,good_key")
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda *a: None)

    used_keys: List[str] = []

    def fake_post(
        self: LLMClient, payload: Dict[str, Any], api_key: str, timeout: float
    ) -> Dict[str, Any]:
        used_keys.append(api_key)
        if api_key == "bad_key":
            raise urllib.error.HTTPError(
                url=self.api_url,
                code=429,
                msg="Too Many Requests",
                hdrs={},  # type: ignore[arg-type]
                fp=io.BytesIO(b"rate limit"),
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": "Thought: ok\n```python\nprint(1)\n```"
                    }
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 15},
        }

    monkeypatch.setattr(LLMClient, "_post_chat", fake_post)

    client = LLMClient(
        model_name="qwen/qwen3-235b-a22b:free",
        provider_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )
    metrics = client.generate(
        messages=[{"role": "user", "content": "Solve task"}],
        step=2,
        stop=["<end_code>"],
    )

    assert used_keys == ["bad_key", "good_key"]
    assert metrics.step == 2
    assert metrics.retries == 1
    assert metrics.input_tokens == 42
    assert metrics.output_tokens == 15
    assert metrics.api_url == "https://openrouter.ai/api/v1"
    assert metrics.model_name == "qwen/qwen3-235b-a22b:free"
    assert "print(1)" in metrics.llm_output
    assert metrics.request_time_ms >= 0.0


def test_llm_client_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1,k2")
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda *a: None)
    monkeypatch.setattr(
        LLMClient,
        "_post_chat",
        lambda self, p, k, t: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    client = LLMClient(
        model_name="llama-3.3-70b-versatile",
        provider_url="https://api.groq.com/openai/v1",
        max_retries=2,
    )
    with pytest.raises(LLMError, match="after 2 retries"):
        client.generate([{"role": "user", "content": "hi"}], step=1)
    assert json.loads(client.rotator.keys[0].replace("k1", '"k1"')) == "k1"


def test_llm_client_respects_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setattr(
        LLMClient, "_sleep_backoff", lambda *a: time.sleep(0.2)
    )
    timeouts: List[float] = []

    def slow_post(
        self: LLMClient, payload: Dict[str, Any], api_key: str, timeout: float
    ) -> Dict[str, Any]:
        timeouts.append(timeout)
        raise RuntimeError("provider down")

    monkeypatch.setattr(LLMClient, "_post_chat", slow_post)
    client = LLMClient(
        model_name="llama-3.3-70b-versatile",
        provider_url="https://api.groq.com/openai/v1",
        max_retries=10,
    )
    with pytest.raises(LLMError):
        client.generate([{"role": "user", "content": "hi"}], timeout=0.5)
    assert len(timeouts) <= 3
    assert all(0 < t <= 0.5 for t in timeouts)


def _rate_limited(retry_after: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1", code=429,
        msg="Too Many Requests",
        hdrs={"retry-after": retry_after},  # type: ignore[arg-type]
        fp=io.BytesIO(b"rate limit"),
    )


@pytest.mark.parametrize("header, deadline_in, expected", [
    ("7", None, 7.0),
    ("60", None, 10.0),
    ("7", 2.0, 2.0),
    ("soon", None, 0.0),
])
def test_retry_after_is_honoured_within_limits(
    header: str, deadline_in: float | None, expected: float
) -> None:
    deadline = (
        None if deadline_in is None else time.perf_counter() + deadline_in)
    wait = LLMClient._retry_after(_rate_limited(header), deadline)
    assert wait == pytest.approx(expected, abs=0.1)


def test_rate_limited_call_waits_before_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    waits: List[float] = []
    monkeypatch.setattr(LLMClient, "_sleep_backoff", staticmethod(
        lambda attempt, wait=0.0: waits.append(wait)))
    calls = iter([_rate_limited("3"), None])

    def fake_post(
        self: LLMClient, payload: Dict[str, Any], api_key: str, timeout: float
    ) -> Dict[str, Any]:
        error = next(calls)
        if error is not None:
            raise error
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(LLMClient, "_post_chat", fake_post)
    client = LLMClient(
        model_name="m", provider_url="https://api.groq.com/openai/v1")
    assert client.generate([{"role": "user", "content": "hi"}]).retries == 1
    assert waits == [pytest.approx(3.0, abs=0.1)]


def test_fallback_answers_when_the_primary_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("MISTRAL_API_KEY", "k2")
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda *a: None)

    def fake_post(
        self: LLMClient, payload: Dict[str, Any], api_key: str, timeout: float
    ) -> Dict[str, Any]:
        if "groq" in self.api_url:
            raise _rate_limited("0")
        return {"choices": [{"message": {"content": "from mistral"}}]}

    monkeypatch.setattr(LLMClient, "_post_chat", fake_post)
    backup = LLMClient(
        model_name="codestral-latest",
        provider_url="https://api.mistral.ai/v1")
    client = LLMClient(
        model_name="qwen", provider_url="https://api.groq.com/openai/v1",
        max_retries=1, fallback=backup)
    metrics = client.generate([{"role": "user", "content": "hi"}])
    assert metrics.llm_output == "from mistral"
    assert metrics.model_name == "codestral-latest"


def test_failed_fallback_reports_both_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("MISTRAL_API_KEY", "k2")
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda *a: None)
    monkeypatch.setattr(
        LLMClient, "_post_chat",
        lambda self, p, k, t: (_ for _ in ()).throw(RuntimeError("down")))
    backup = LLMClient(
        model_name="codestral-latest",
        provider_url="https://api.mistral.ai/v1",
        max_retries=0)
    client = LLMClient(
        model_name="qwen", provider_url="https://api.groq.com/openai/v1",
        max_retries=0, fallback=backup)
    with pytest.raises(LLMError, match="fallback codestral-latest failed"):
        client.generate([{"role": "user", "content": "hi"}])
