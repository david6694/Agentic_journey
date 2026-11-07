"""CLI entry point for the MBPP agent (`uv run python -m agent_mbpp`)."""
import argparse
import os
from pathlib import Path
import shlex
import sys
import time

from smith.agent.loop import AgentLimits, run_agent
from smith.agent.prompts import (
    build_system_prompt,
    mbpp_setup_code,
    mbpp_task_prompt,
)
from smith.llm.client import LLMClient
from smith.models import MBPPTaskInput, SolutionOutput
from smith.sandbox.config import SmithSandboxConfig, load_config
from smith.sandbox.mcp_client import McpClient
from smith.sandbox.sandbox import Sandbox


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse MBPP agent command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="agent_mbpp",
        description="Solve an MBPP task using the Agent Smith framework.",
    )
    parser.add_argument(
        "--task-file",
        required=True,
        type=Path,
        help="Path to the input MBPP task JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the SolutionOutput JSON file will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier (e.g., 'qwen/qwen3-235b-a22b:free').",
    )
    parser.add_argument(
        "--provider-url",
        default=None,
        help="LLM API base URL (e.g., 'https://openrouter.ai/api/v1').",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override maximum agent iterations (default: 10).",
    )
    parser.add_argument(
        "--sandbox-config",
        type=Path,
        default=None,
        help="Optional path to a sandbox configuration JSON file.",
    )
    return parser.parse_args(argv)


# Best MBPP model in BENCHMARK_REPORT.md. Groq's per-minute quota makes the
# fallback, the default model of configs/providers.json, necessary.
MBPP_MODEL = ("qwen/qwen3.8-27b", "https://api.groq.com/openai/v1")


def build_llm(
    model_name: str | None, provider_url: str | None
) -> LLMClient:
    """The model asked for, else MBPP_MODEL with a fallback client."""
    if (model_name or provider_url or os.environ.get("LLM_MODEL_NAME")
            or os.environ.get("LLM_PROVIDER_URL")):
        return LLMClient(model_name=model_name, provider_url=provider_url)
    return LLMClient(*MBPP_MODEL, fallback=LLMClient())


def _write_output(path: Path, solution: SolutionOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(solution.model_dump_json(indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Load an MBPP task, run the agent loop, and write solution.json."""
    args = parse_args(argv)
    start = time.monotonic()

    try:
        raw_task = args.task_file.read_text(encoding="utf-8")
        task = MBPPTaskInput.model_validate_json(raw_task)
    except Exception as exc:
        fallback = SolutionOutput(
            task_id="unknown",
            benchmark="mbpp",
            success=False,
            solution="",
            iterations=0,
            total_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_time_seconds=time.monotonic() - start,
            error=f"Failed to load task file: {exc}",
        )
        _write_output(args.output, fallback)
        return 0

    limits = AgentLimits.mbpp()
    if args.max_iterations is not None:
        limits.max_iterations = max(1, min(10, args.max_iterations))

    repo_root = Path(__file__).resolve().parent.parent
    mcp_script = repo_root / "mcp_tools_mbpp.py"
    stdio_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(mcp_script))}"

    try:
        # The moulinette kills the agent at 120s: one block must stay short.
        config = (
            load_config(args.sandbox_config)
            if args.sandbox_config
            else SmithSandboxConfig(
                max_execution_time_seconds=10, mcp_call_timeout_seconds=15
            )
        )
        llm = build_llm(args.model_name, args.provider_url)
        with McpClient(
            "stdio", stdio_cmd, call_timeout=config.mcp_call_timeout_seconds
        ) as mcp_client:
            with Sandbox(config, mcp_client) as sandbox:
                system_prompt = build_system_prompt(sandbox.manual, "mbpp")
                task_prompt = mbpp_task_prompt(task)
                setup_code = mbpp_setup_code(task)
                solution = run_agent(
                    llm=llm,
                    sandbox=sandbox,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    limits=limits,
                    task_id=str(task.task_id),
                    benchmark="mbpp",
                    setup_code=setup_code,
                    started_at=start,
                )
    except Exception as exc:
        solution = SolutionOutput(
            task_id=str(task.task_id),
            benchmark="mbpp",
            success=False,
            solution="",
            iterations=0,
            total_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_time_seconds=time.monotonic() - start,
            error=f"Agent execution failed: {type(exc).__name__}: {exc}",
        )

    _write_output(args.output, solution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
