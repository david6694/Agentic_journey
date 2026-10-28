"""Parent side of the sandbox.

The Sandbox owns a long-lived worker process (see worker.py) and talks
to it through a pipe. Untrusted code never runs in this process, which
lets us kill it on timeout or memory overuse at any moment. The MCP
client lives here too: tool calls made by the sandboxed code are
relayed through the pipe, so the worker itself needs no network.

Every message from the worker is untrusted: it is decoded as JSON and
its fields are type-checked before use.
"""

import multiprocessing
import subprocess
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from smith.sandbox.config import SmithSandboxConfig
from smith.sandbox.manual import build_manual
from smith.sandbox.mcp_client import McpClient
from smith.sandbox.results import ExecutionResult
from smith.sandbox.worker import recv, send, worker_main

POLL_INTERVAL_SECONDS = 0.05
MEMORY_CHECK_INTERVAL_SECONDS = 0.25
WORKER_START_TIMEOUT_SECONDS = 30

# A handler receives the call arguments and returns (ok, text).
Handler = Callable[[dict[str, Any]], tuple[bool, str]]


class SandboxError(Exception):
    """Raised when the worker process cannot be started."""


def process_rss_mb(pid: int) -> float | None:
    """Resident memory of a process in MB (Linux /proc, else `ps`)."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False)
        return int(completed.stdout.strip()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


class Sandbox:
    """Secure execution environment for LLM-generated Python code.

    Typical use:

        with Sandbox(config, mcp_client) as sandbox:
            result = sandbox.execute("print(read_file('/testbed/a.py'))")
            observation = result.to_observation()
    """

    def __init__(
        self,
        config: SmithSandboxConfig | None = None,
        mcp: McpClient | None = None,
    ) -> None:
        """Prepare the sandbox. The worker starts on first use.

        Args:
            config: Limits and allowlists. Defaults to the subject's.
            mcp: Connected MCP client whose tools are exposed, if any.
        """
        self.config = config or SmithSandboxConfig()
        self._mcp = mcp
        self._handlers: dict[str, Handler] = {}
        self._specs: list[dict[str, Any]] = []
        self._process: BaseProcess | None = None
        self._conn: Connection | None = None
        self._baseline_rss_mb = 0.0
        self._register_mcp()

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        """True while the worker process is alive."""
        return self._process is not None and self._process.is_alive()

    @property
    def manual(self) -> str:
        """Manual describing this sandbox and its tools for the LLM."""
        mcp = self._mcp
        return build_manual(
            self.config,
            tools=mcp.tools if mcp else (),
            has_resources=bool(mcp and mcp.has_resources),
            has_prompts=bool(mcp and mcp.has_prompts),
        )

    @property
    def tool_names(self) -> list[str]:
        """Names callable from the sandboxed code (final_answer aside)."""
        return [spec["name"] for spec in self._specs]

    def _add_callable(
        self, name: str, params: list[str], handler: Handler
    ) -> None:
        if name in ("final_answer", "ToolError") or name in self._handlers:
            return
        self._handlers[name] = handler
        self._specs.append({"name": name, "params": params})

    def _register_mcp(self) -> None:
        """Expose MCP tools, resources and prompts as sandbox callables."""
        mcp = self._mcp
        if mcp is None:
            return
        for tool in mcp.tools:
            def call(args: dict[str, Any], name: str = tool.name
                     ) -> tuple[bool, str]:
                return mcp.call_tool(name, args)
            self._add_callable(tool.python_name, tool.param_names, call)
        if mcp.has_resources:
            self._add_callable(
                "list_mcp_resources", [],
                lambda args: (True, mcp.list_resources()))
            self._add_callable(
                "read_mcp_resource", ["uri"],
                lambda args: (True, mcp.read_resource(args["uri"])))
        if mcp.has_prompts:
            self._add_callable(
                "list_mcp_prompts", [],
                lambda args: (True, mcp.list_prompts()))
            self._add_callable(
                "get_mcp_prompt", ["name", "arguments"],
                lambda args: (True, mcp.get_prompt(
                    args["name"], args.get("arguments"))))

    def start(self) -> None:
        """Start the worker process if it is not running.

        Raises:
            SandboxError: The worker crashed or did not boot in time.
        """
        if self.is_running:
            return
        context = multiprocessing.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        process = context.Process(
            target=worker_main,
            args=(child_conn, self.config.model_dump(), self._specs),
            name="smith-sandbox-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process, self._conn = process, parent_conn
        message: dict[str, Any] = {}
        try:
            if parent_conn.poll(WORKER_START_TIMEOUT_SECONDS):
                message = recv(parent_conn)
        except (EOFError, OSError, ValueError):
            pass
        if message.get("type") != "ready":
            self._stop_worker(graceful=False)
            raise SandboxError(
                "sandbox worker failed to start "
                f"(exit code {process.exitcode}):\n"
                f"{message.get('error', 'no answer')}")
        if process.pid is not None:
            self._baseline_rss_mb = process_rss_mb(process.pid) or 0.0

    def close(self) -> None:
        """Stop the worker process."""
        self._stop_worker(graceful=True)

    def _stop_worker(self, graceful: bool) -> None:
        process, conn = self._process, self._conn
        self._process, self._conn = None, None
        if process is None:
            return
        if graceful and conn is not None and process.is_alive():
            try:
                send(conn, {"type": "shutdown"})
                process.join(timeout=1)
            except (OSError, ValueError):
                pass
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if conn is not None:
            conn.close()

    def _restart_worker(self) -> None:
        self._stop_worker(graceful=False)
        self.start()

    def _memory_exceeded(self) -> bool:
        """Parent-side watchdog, needed where RLIMIT_AS is not enforced."""
        process = self._process
        if process is None or process.pid is None:
            return False
        rss = process_rss_mb(process.pid)
        budget = self._baseline_rss_mb + self.config.max_memory_mb
        return rss is not None and rss > budget

    def _dispatch(
        self, name: object, args: object
    ) -> tuple[dict[str, Any], str | None]:
        """Run a tool call from the worker. Returns (reply, note)."""
        handler = self._handlers.get(name) if isinstance(name, str) else None
        if handler is None or not isinstance(args, dict):
            return {"ok": False, "result": f"invalid tool call '{name}'"}, \
                None
        try:
            ok, text = handler(args)
        except Exception as exc:
            return {"ok": False, "result": f"{type(exc).__name__}: {exc}"}, \
                None
        limit = self.config.max_tool_output_chars
        note = None
        if len(text) > limit:
            note = (f"Output of tool '{name}' was truncated from "
                    f"{len(text)} to {limit} characters.")
            text = (text[:limit]
                    + f"\n... [truncated {len(text) - limit} characters]")
        return {"ok": ok, "result": text}, note

    def execute(self, code: str) -> ExecutionResult:
        """Run code in the worker and collect everything that happened.

        Time spent inside MCP tool calls is not counted in the timeout.

        Raises:
            SystemExit: The code raised SystemExit.
            KeyboardInterrupt: The code raised it, or Ctrl+C was pressed
                (the worker is then restarted).
            SandboxError: The worker could not be (re)started.
        """
        if not self.is_running:
            self.start()
        conn = self._conn
        assert conn is not None
        result = ExecutionResult()
        output: list[str] = []
        output_size = 0
        limit = self.config.max_output_chars
        remaining = float(self.config.max_execution_time_seconds)
        next_memory_check = 0.0
        done: dict[str, Any] | None = None
        try:
            send(conn, {"type": "exec", "code": code})
            while True:
                started = time.monotonic()
                message = None
                if conn.poll(max(0.0, min(remaining, POLL_INTERVAL_SECONDS))):
                    message = recv(conn)
                remaining -= time.monotonic() - started
                if message is not None:
                    kind = message.get("type")
                    if kind == "output":
                        # The worker caps its output too; never trust it.
                        data = str(message.get("data", ""))[
                            :max(0, limit - output_size)]
                        output.append(data)
                        output_size += len(data)
                    elif kind == "call":
                        reply, note = self._dispatch(
                            message.get("name"), message.get("args"))
                        if note:
                            result.notes.append(note)
                        send(conn, reply)
                    elif kind == "done":
                        done = message
                        break
                elif not self.is_running:
                    raise EOFError
                if remaining <= 0:
                    result.timed_out = True
                    result.error = (
                        "TimeoutError: execution exceeded "
                        f"{self.config.max_execution_time_seconds}s")
                    break
                now = time.monotonic()
                if now >= next_memory_check:
                    next_memory_check = now + MEMORY_CHECK_INTERVAL_SECONDS
                    if self._memory_exceeded():
                        result.memory_exceeded = True
                        result.error = (
                            "MemoryError: execution exceeded "
                            f"{self.config.max_memory_mb} MB")
                        break
        except (EOFError, OSError, ValueError):
            process = self._process
            exit_code = process.exitcode if process else None
            result.error = (
                "SandboxCrash: the worker process died or broke the "
                f"protocol (exit code {exit_code}), possibly out of memory")
        except KeyboardInterrupt:
            self._restart_worker()
            raise

        result.output = "".join(output)
        result.output_truncated = output_size >= limit
        if done is None:
            self._restart_worker()
            result.worker_restarted = True
            return result
        return self._finish(result, done)

    def _finish(
        self, result: ExecutionResult, done: dict[str, Any]
    ) -> ExecutionResult:
        """Merge the worker's 'done' message into the result."""
        if done.get("control") == "SystemExit":
            exit_code = done.get("exit_code")
            raise SystemExit(
                exit_code if isinstance(exit_code, (int, str)) else None)
        if done.get("control") == "KeyboardInterrupt":
            raise KeyboardInterrupt
        result.error = _optional_str(done.get("error"))
        result.memory_exceeded = bool(done.get("memory_error"))
        result.output_truncated = (
            result.output_truncated or bool(done.get("truncated")))
        result.final_answer_called = bool(done.get("final_called"))
        result.final_answer = _optional_str(done.get("final_answer"))
        return result
