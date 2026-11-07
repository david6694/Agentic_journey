"""End-to-end tests for the MBPP agent CLI (`python -m agent_mbpp`)."""
import json
from pathlib import Path
from typing import Dict, List

import pytest

from agent_mbpp.__main__ import build_llm, main
from smith.llm.client import LLMClient
from smith.models import MBPPTaskInput, SolutionOutput, StepMetrics


def test_agent_mbpp_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--task-file" in captured.out
    assert "--output" in captured.out
    assert "--model-name" in captured.out
    assert "--provider-url" in captured.out


def test_agent_mbpp_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy_key")

    task = MBPPTaskInput(
        task_id=42,
        task_definition="Write a function to add two numbers.",
        function_definition="def add_nums(a, b):",
        test_imports=[],
        test_list=[
            "assert add_nums(1, 2) == 3",
            "assert add_nums(-1, 5) == 4",
        ],
    )
    task_file = tmp_path / "task.json"
    out_file = tmp_path / "solution.json"
    task_file.write_text(task.model_dump_json(), encoding="utf-8")

    def fake_generate(
        self: LLMClient,
        messages: List[Dict[str, str]],
        step: int = 1,
        stop: List[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> StepMetrics:
        llm_reply = (
            "Thought: Test and submit my implementation.\n"
            "```python\n"
            "import json\n"
            "code = '''def add_nums(a, b):\n    return a + b\n'''\n"
            "res = json.loads(run_tests(code, test_list))\n"
            "print(res)\n"
            "if res['success']:\n"
            "    final_answer(code)\n"
            "```<end_code>"
        )
        return StepMetrics(
            step=step,
            input_tokens=120,
            output_tokens=60,
            request_time_ms=50.0,
            api_url=self.api_url,
            model_name=self.model_name,
            llm_output=llm_reply,
            retries=0,
        )

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    exit_code = main(
        [
            "--task-file",
            str(task_file),
            "--output",
            str(out_file),
            "--model-name",
            "test/model",
            "--provider-url",
            "https://openrouter.ai/api/v1",
        ]
    )
    assert exit_code == 0
    assert out_file.is_file()

    sol = SolutionOutput.model_validate(
        json.loads(out_file.read_text(encoding="utf-8"))
    )
    assert sol.task_id == "42"
    assert sol.benchmark == "mbpp"
    assert sol.success is True
    assert "def add_nums(a, b):" in sol.solution
    assert sol.iterations == 1
    assert sol.total_input_tokens == 120
    assert sol.total_output_tokens == 60
    assert len(sol.steps) == 1
    assert "run_tests" in sol.steps[0].sandbox_input
    assert "All 2 tests passed" in sol.steps[0].sandbox_output
    assert "run_tests" in sol.system_prompt
    assert "Limits per execution: 10s" in sol.system_prompt


def test_default_model_is_qwen_with_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("LLM_MODEL_NAME", "LLM_PROVIDER_URL"):
        monkeypatch.delenv(name, raising=False)
    llm = build_llm(None, None)
    assert llm.model_name == "qwen/qwen3.8-27b"
    assert llm.fallback is not None and llm.fallback.fallback is None
    assert "groq" not in llm.fallback.api_url


def test_requested_model_has_no_fallback() -> None:
    llm = build_llm("codestral-latest", "https://api.mistral.ai/v1")
    assert llm.model_name == "codestral-latest" and llm.fallback is None
