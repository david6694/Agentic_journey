"""End-to-end and Docker lifecycle tests for `python -m agent_swebench`."""
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple, cast

import pytest

from agent_swebench.__main__ import _submit_unfinished_work, main
from agent_swebench.docker_env import SWEBenchContainer
from smith.llm.client import LLMClient
from smith.models import SolutionOutput, StepMetrics, SWEBenchTaskInput
from smith.sandbox.mcp_client import McpClient, McpConnectionError


def _init_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo_dir),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo_dir),
        check=True,
    )
    (repo_dir / "buggy.py").write_text(
        "def divide(a, b):\n    return a * b\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=str(repo_dir), check=True
    )


def test_agent_swebench_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--task-file" in captured.out
    assert "--output" in captured.out
    assert "--model-name" in captured.out
    assert "--provider-url" in captured.out


def test_agent_swebench_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy_key")
    fake_testbed = tmp_path / "testbed"
    _init_git_repo(fake_testbed)
    cleaned_up = {"called": False}

    def fake_start(self: SWEBenchContainer) -> None:
        self.testbed_path = fake_testbed

    def fake_cleanup(self: SWEBenchContainer) -> None:
        cleaned_up["called"] = True

    def fake_configure_env(
        self: SWEBenchContainer, setter: object = None
    ) -> None:
        os.environ["TESTBED_PATH"] = str(fake_testbed)
        os.environ.pop("SWE_CONTAINER_ID", None)
        os.environ["EVAL_SCRIPT"] = (
            f"{sys.executable} -c "
            "\"from buggy import divide; assert divide(6, 2) == 3\""
        )

    monkeypatch.setattr(SWEBenchContainer, "start", fake_start)
    monkeypatch.setattr(SWEBenchContainer, "cleanup", fake_cleanup)
    monkeypatch.setattr(SWEBenchContainer, "configure_env", fake_configure_env)

    task = SWEBenchTaskInput(
        instance_id="demo__repo-123",
        problem_statement="divide(a, b) multiplies instead of dividing.",
        docker_image="swebench/sweb.eval.x86_64.demo_123:latest",
        eval_script="pytest",
        repo="demo/repo",
    )
    task_file = tmp_path / "swe_task.json"
    out_file = tmp_path / "swe_solution.json"
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
        if step == 1:
            code = (
                "Thought: Fix buggy.py and run tests.\n"
                "```python\n"
                "print(edit_file('/testbed/buggy.py', 'return a * b', "
                "'return a / b'))\n"
                "print(run_tests())\n"
                "```<end_code>"
            )
        else:
            code = (
                "Thought: Tests passed, submit patch.\n"
                "```python\n"
                "final_answer(get_patch())\n"
                "```<end_code>"
            )
        return StepMetrics(
            step=step,
            input_tokens=200,
            output_tokens=80,
            request_time_ms=40.0,
            api_url=self.api_url,
            model_name=self.model_name,
            llm_output=code,
            retries=0,
        )

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    rc = main(
        [
            "--task-file",
            str(task_file),
            "--output",
            str(out_file),
            "--model-name",
            "test/swe-model",
            "--provider-url",
            "https://openrouter.ai/api/v1",
        ]
    )
    assert rc == 0
    assert cleaned_up["called"] is True
    assert out_file.is_file()

    sol = SolutionOutput.model_validate(
        json.loads(out_file.read_text(encoding="utf-8"))
    )
    assert sol.task_id == "demo__repo-123"
    assert sol.benchmark == "swebench"
    assert sol.success is True
    assert "-    return a * b" in sol.solution
    assert "+    return a / b" in sol.solution
    assert sol.iterations == 2
    # Writing to the local copy directly would never reach the patch.
    assert str(fake_testbed) not in sol.system_prompt


class _PatchClient:
    def __init__(self, reply: Tuple[bool, str] | None) -> None:
        self.reply = reply

    def call_tool(self, name: str, arguments: object) -> Tuple[bool, str]:
        assert name == "get_patch"
        if self.reply is None:
            raise McpConnectionError("server gone")
        return self.reply


def _solution(success: bool, answer: str = "") -> SolutionOutput:
    return SolutionOutput(
        task_id="demo", benchmark="swebench", success=success,
        solution=answer, iterations=30, total_requests=30,
        total_input_tokens=1, total_output_tokens=1, total_time_seconds=1.0,
        error=None if success else "no final_answer within 30 iterations",
    )


def _submit(sol: SolutionOutput, reply: Tuple[bool, str] | None) -> None:
    client = cast(McpClient, _PatchClient(reply))
    _submit_unfinished_work(sol, client)


def test_unfinished_run_submits_the_repository_patch() -> None:
    sol = _solution(False)
    _submit(sol, (True, "diff --git a/x b/x\n"))
    assert sol.solution == "diff --git a/x b/x\n"
    assert sol.success is False and sol.error is not None


def test_final_answer_is_left_untouched() -> None:
    sol = _solution(True, "diff --git a/agent b/agent\n")
    _submit(sol, (True, "diff --git a/other b/other\n"))
    assert sol.solution == "diff --git a/agent b/agent\n"


@pytest.mark.parametrize("reply", [
    (True, ""), (True, "Error running git diff: x"), (False, "boom"), None,
])
def test_no_usable_patch_keeps_the_empty_solution(
    reply: Tuple[bool, str] | None,
) -> None:
    sol = _solution(False)
    _submit(sol, reply)
    assert sol.solution == ""
