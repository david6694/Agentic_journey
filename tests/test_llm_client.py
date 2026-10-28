"""Tests for smith.llm (KeyRotator and LLMClient)."""
import io
import json
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


def test_llm_client_rotation_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad_key,good_key")
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda self, a: None)

    used_keys: List[str] = []

    def fake_post(
        self: LLMClient, payload: Dict[str, Any], api_key: str
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
    monkeypatch.setattr(LLMClient, "_sleep_backoff", lambda self, a: None)
    monkeypatch.setattr(
        LLMClient,
        "_post_chat",
        lambda self, p, k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    client = LLMClient(
        model_name="llama-3.3-70b-versatile",
        provider_url="https://api.groq.com/openai/v1",
        max_retries=2,
    )
    with pytest.raises(LLMError, match="after 2 retries"):
        client.generate([{"role": "user", "content": "hi"}], step=1)
    assert json.loads(client.rotator.keys[0].replace("k1", '"k1"')) == "k1"
