"""Agent loop with a scripted LLM and a real sandbox."""

from collections.abc import Iterator
import time

import pytest

from smith.agent.loop import AgentLimits, run_agent
from smith.models import SolutionOutput, StepMetrics
from smith.sandbox.config import SmithSandboxConfig
from smith.sandbox.results import ExecutionResult
from smith.sandbox.sandbox import Sandbox, SandboxError


class FakeLLM:
    """Returns scripted answers; raises when the script is exhausted.

    Input tokens are counted like a real tokenizer would (~4 chars each).
    """

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[list[dict[str, str]]] = []
        self.max_tokens: list[int | None] = []

    def generate(self, messages: list[dict[str, str]], *, step: int,
                 stop: list[str] | None, max_tokens: int | None,
                 timeout: float | None) -> StepMetrics:
        assert timeout is not None
        assert 0 < timeout <= AgentLimits.mbpp().max_seconds
        self.calls.append(list(messages))
        self.max_tokens.append(max_tokens)
        if not self.answers:
            raise RuntimeError("provider down")
        size = sum(len(m["content"]) for m in messages)
        return StepMetrics(
            step=step, llm_output=self.answers.pop(0),
            input_tokens=size // 4, output_tokens=50, request_time_ms=5.0,
            retries=1, api_url="https://fake/v1", model_name="fake-model")


@pytest.fixture
def sandbox() -> Iterator[Sandbox]:
    with Sandbox(SmithSandboxConfig(allowed_directories=[])) as box:
        yield box


def _run(llm: FakeLLM, sandbox: Sandbox, limits: AgentLimits | None = None,
         setup_code: str = "", system: str = "system") -> SolutionOutput:
    output = run_agent(
        llm, sandbox, system, "task", limits or AgentLimits.mbpp(),
        task_id="1", benchmark="mbpp", setup_code=setup_code)
    return SolutionOutput.model_validate(output.model_dump())


def test_solves_in_two_turns(sandbox: Sandbox) -> None:
    llm = FakeLLM([
        "Thought: try\n```python\nx = 2 + 2\nprint(x)\n```",
        "Thought: done\n```python\nfinal_answer(f'def f(): return {x}')\n```",
    ])
    output = _run(llm, sandbox)
    assert output.success and output.error is None
    assert output.solution == "def f(): return 4"
    assert output.iterations == 2
    assert output.total_requests == 4  # 2 calls, 1 retry each
    assert output.total_output_tokens == 100
    assert output.steps[0].sandbox_output == "4"
    assert "Observation:\n4" in llm.calls[1][-1]["content"]


def test_missing_code_gets_feedback(sandbox: Sandbox) -> None:
    llm = FakeLLM(["The answer is 4.", "```python\nfinal_answer('4')\n```"])
    output = _run(llm, sandbox)
    assert output.success
    assert "No code block found" in llm.calls[1][-1]["content"]
    assert output.steps[0].sandbox_input == ""


def test_iteration_limit(sandbox: Sandbox) -> None:
    llm = FakeLLM([f"```python\nprint({i})\n```" for i in range(5)])
    limits = AgentLimits.mbpp().model_copy(update={"max_iterations": 3})
    output = _run(llm, sandbox, limits)
    assert not output.success and output.iterations == 3
    assert output.error is not None and "3 iterations" in output.error


def test_input_budget_stops_before_overflow(sandbox: Sandbox) -> None:
    llm = FakeLLM(["```python\nprint(1)\n```"] * 10)
    output = _run(llm, sandbox, system="x" * 8_000)  # ~2,000 tokens a call
    assert output.total_input_tokens <= 6_000
    assert output.error is not None and "input token" in output.error


def test_llm_failure_is_reported(sandbox: Sandbox) -> None:
    output = _run(FakeLLM([]), sandbox)
    assert not output.success
    assert output.error is not None and "provider down" in output.error


def test_setup_code_defines_variables(sandbox: Sandbox) -> None:
    llm = FakeLLM(["```python\nfinal_answer(str(test_list))\n```"])
    output = _run(llm, sandbox, setup_code="test_list = ['assert 1']")
    assert output.solution == "['assert 1']"


def test_setup_code_survives_restart() -> None:
    config = SmithSandboxConfig(allowed_directories=[],
                                max_execution_time_seconds=1)
    llm = FakeLLM(["```python\nwhile True: pass\n```",
                   "```python\nfinal_answer(str(test_list))\n```"])
    with Sandbox(config) as box:
        output = _run(llm, box, setup_code="test_list = ['assert 1']")
    assert output.solution == "['assert 1']"


def test_output_is_capped_per_turn_then_by_what_is_left(
    sandbox: Sandbox,
) -> None:
    limits = AgentLimits.mbpp()
    limits.max_output_tokens = 440
    llm = FakeLLM(["```python\nprint(1)\n```",
                   "```python\nfinal_answer('done')\n```"])
    _run(llm, sandbox, limits)
    assert llm.max_tokens == [limits.max_turn_output_tokens, 390]


def test_time_before_the_loop_counts(sandbox: Sandbox) -> None:
    limits = AgentLimits.mbpp()
    llm = FakeLLM(["```python\nfinal_answer('never asked')\n```"])
    output = run_agent(
        llm, sandbox, "system", "task", limits, task_id="1",
        benchmark="mbpp", started_at=time.monotonic() - limits.max_seconds)
    assert not llm.calls and not output.success
    assert output.error is not None and "time budget" in output.error
    assert output.total_time_seconds >= limits.max_seconds


class BrokenSandbox:
    def execute(self, code: str) -> ExecutionResult:
        raise SandboxError("worker failed to start")


def test_sandbox_failure_is_reported() -> None:
    llm = FakeLLM(["```python\nprint(1)\n```"])
    output = _run(llm, BrokenSandbox())  # type: ignore[arg-type]
    assert not output.success and output.iterations == 1
    assert output.error is not None and "sandbox failed" in output.error
