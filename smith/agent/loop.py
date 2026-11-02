"""The agent loop: Thought -> Code -> Observation until final_answer."""

import time

from pydantic import BaseModel

from smith.agent.extraction import extract_code
from smith.llm.base import ChatModel
from smith.models import SolutionOutput
from smith.sandbox.sandbox import Sandbox, SandboxError

STOP_SEQUENCES = ["<end_code>", "Observation:"]
# Code tokenizes at ~3-4 chars per token; 3 keeps the estimate pessimistic.
CHARS_PER_TOKEN = 3
MIN_OUTPUT_TOKENS = 64


class AgentLimits(BaseModel):
    """Budget of one task. Token limits are cumulative over all turns."""

    max_iterations: int
    max_input_tokens: int
    max_output_tokens: int
    max_turn_output_tokens: int
    max_seconds: float
    max_observation_chars: int

    @classmethod
    def mbpp(cls) -> "AgentLimits":
        """Subject limits, with 30s kept for the last block and startup.

        One rambling answer must not consume the whole output budget.
        """
        return cls(max_iterations=10, max_input_tokens=6_000,
                   max_output_tokens=1_500, max_turn_output_tokens=400,
                   max_seconds=90, max_observation_chars=1_000)

    @classmethod
    def swebench(cls) -> "AgentLimits":
        """Subject limits, with 150s kept for a last run_tests and cleanup."""
        return cls(max_iterations=30, max_input_tokens=300_000,
                   max_output_tokens=10_000, max_turn_output_tokens=10_000,
                   max_seconds=750, max_observation_chars=8_000)


def _shorten(text: str, limit: int) -> str:
    """Keep the head and the tail, where errors usually are."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (f"{text[:half]}\n... [{len(text) - limit} characters cut] ...\n"
            f"{text[-half:]}")


def _run_setup(sandbox: Sandbox, setup_code: str) -> str | None:
    """Run the setup code; return an error message if it failed."""
    if not setup_code:
        return None
    result = sandbox.execute(setup_code)
    return f"setup code failed: {result.error}" if result.error else None


def _out_of_budget(
    output: SolutionOutput, messages: list[dict[str, str]],
    limits: AgentLimits, elapsed: float,
) -> str | None:
    """Reason to stop before the next LLM call, or None."""
    if elapsed >= limits.max_seconds:
        return f"time budget of {limits.max_seconds:.0f}s exhausted"
    estimate = sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN
    if output.total_input_tokens + estimate > limits.max_input_tokens:
        return "input token budget would be exceeded by the next call"
    if limits.max_output_tokens - output.total_output_tokens \
            < MIN_OUTPUT_TOKENS:
        return "output token budget exhausted"
    return None


def run_agent(
    llm: ChatModel,
    sandbox: Sandbox,
    system_prompt: str,
    task_prompt: str,
    limits: AgentLimits,
    task_id: str,
    benchmark: str,
    setup_code: str = "",
    started_at: float | None = None,
) -> SolutionOutput:
    """Solve one task and return the SolutionOutput the moulinette reads.

    Args:
        llm: Client used for every turn.
        sandbox: Started sandbox, with the task's MCP tools.
        system_prompt: Full system prompt (stored for provenance).
        task_prompt: First user message.
        limits: Iteration, token and time budget.
        task_id: MBPP task id or SWE-bench instance id.
        benchmark: 'mbpp' or 'swebench'.
        setup_code: Run once in the sandbox before the first turn.
        started_at: time.monotonic() when the agent process started, so
            that its startup counts in the budget and in total_time_seconds.
    """
    start = time.monotonic() if started_at is None else started_at
    output = SolutionOutput(
        task_id=task_id, benchmark=benchmark, success=False, solution="",
        iterations=0, total_requests=0, total_input_tokens=0,
        total_output_tokens=0, total_time_seconds=0.0,
        system_prompt=system_prompt)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}]
    try:
        output.error = _run_setup(sandbox, setup_code)
    except SandboxError as exc:
        output.error = f"sandbox failed: {exc}"
    for step in range(1, limits.max_iterations + 1):
        if output.error:
            break
        elapsed = time.monotonic() - start
        output.error = _out_of_budget(output, messages, limits, elapsed)
        if output.error:
            break
        try:
            metrics = llm.generate(
                messages, step=step, stop=STOP_SEQUENCES,
                max_tokens=min(
                    limits.max_turn_output_tokens,
                    limits.max_output_tokens - output.total_output_tokens),
                timeout=limits.max_seconds - elapsed)
        except Exception as exc:
            output.error = f"LLM request failed: {type(exc).__name__}: {exc}"
            break
        code, note = extract_code(metrics.llm_output)
        try:
            result = sandbox.execute(code) if code else None
            if result and result.worker_restarted:
                # A restarted worker lost the variables defined by setup.
                output.error = _run_setup(sandbox, setup_code)
        except SandboxError as exc:
            result, output.error = None, f"sandbox failed: {exc}"
        observation = _shorten(
            result.to_observation(), limits.max_observation_chars
        ) if result else ""
        metrics.sandbox_input = code or ""
        metrics.sandbox_output = observation
        output.steps.append(metrics)
        output.iterations = step
        output.total_requests += 1 + metrics.retries
        output.total_input_tokens += metrics.input_tokens
        output.total_output_tokens += metrics.output_tokens
        if result and result.final_answer_called:
            output.success = True
            output.solution = result.final_answer or ""
            break
        if output.error:
            break
        feedback = "\n".join(p for p in (note, observation) if p)
        messages += [{"role": "assistant", "content": metrics.llm_output},
                     {"role": "user", "content": f"Observation:\n{feedback}"}]
    else:
        output.error = (f"no final_answer within {limits.max_iterations} "
                        "iterations")
    output.total_time_seconds = time.monotonic() - start
    return output
