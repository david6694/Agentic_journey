"""Sandbox worker: the child process that runs untrusted code.

Protocol: JSON objects over a multiprocessing Pipe. JSON, never pickle:
unpickling can run code, and the parent must treat everything coming
from the worker as untrusted.

    parent -> worker  {"type": "exec", "code": str}
                      {"type": "shutdown"}
                      {"ok": bool, "result": str}   (reply to "call")
    worker -> parent  {"type": "ready"}
                      {"type": "fatal", "error": str}
                      {"type": "output", "data": str}
                      {"type": "call", "name": str, "args": dict}
                      {"type": "done", ...}          (see _execute)

Only the standard library is imported here, to keep the worker light.
"""

import ast
import importlib
import io
import json
import linecache
import os
import signal
import sys
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from smith.sandbox import security

Message = dict[str, Any]
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def send(conn: Connection, message: Message) -> None:
    """Send one JSON message."""
    conn.send_bytes(json.dumps(message).encode())


def recv(conn: Connection) -> Message:
    """Receive one JSON message.

    Raises:
        EOFError: The other side closed the pipe.
        OSError: The message is larger than MAX_MESSAGE_BYTES.
        ValueError: The message is not a JSON object.
    """
    message = json.loads(conn.recv_bytes(MAX_MESSAGE_BYTES))
    if not isinstance(message, dict):
        raise ValueError("protocol message is not a JSON object")
    return message


class ToolError(Exception):
    """Raised in sandbox code when a tool call fails."""


# Display "ToolError: ..." instead of "smith.sandbox.worker.ToolError".
ToolError.__module__ = "builtins"


class _FinalAnswerSignal(BaseException):
    """Stops the current execution once final_answer() is called.

    BaseException so that a bare `except Exception` in the sandboxed
    code cannot swallow it.
    """


class _PipeWriter(io.TextIOBase):
    """stdout/stderr replacement that streams text to the parent.

    Streaming (instead of buffering) keeps the output that was printed
    before a timeout kill. Text beyond the limit is dropped and flagged.
    """

    def __init__(self, conn: Connection, limit: int) -> None:
        super().__init__()
        self._conn = conn
        self._limit = limit
        self.written = 0
        self.truncated = False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {text!r}")
        remaining = self._limit - self.written
        if remaining > 0 and text:
            chunk = text[:remaining]
            send(self._conn, {"type": "output", "data": chunk})
            self.written += len(chunk)
        if len(text) > max(remaining, 0):
            self.truncated = True
        return len(text)

    def reset(self) -> None:
        """Start counting again for a new execution."""
        self.written = 0
        self.truncated = False


def _make_tool_stub(
    conn: Connection, name: str, params: list[str]
) -> Callable[..., Any]:
    """Build the Python function that forwards a tool call to the parent.

    Positional arguments are mapped to parameter names in the order
    published in the manual (required first, then optional).
    """
    def stub(*args: Any, **kwargs: Any) -> Any:
        if len(args) > len(params):
            raise TypeError(
                f"{name}() takes at most {len(params)} positional "
                f"arguments ({len(args)} given)"
            )
        call_args = dict(zip(params, args))
        for key, value in kwargs.items():
            if key in call_args:
                raise TypeError(f"{name}() got multiple values for '{key}'")
            call_args[key] = value
        try:
            send(conn, {"type": "call", "name": name, "args": call_args})
        except TypeError as exc:
            raise TypeError(
                f"{name}(): arguments must be plain data (str, int, "
                f"list, dict...): {exc}"
            ) from None
        reply = recv(conn)
        if not reply["ok"]:
            raise ToolError(reply["result"])
        return reply["result"]

    stub.__name__ = name
    return stub


def _format_exception(exc: BaseException) -> str:
    """Format a traceback keeping only the frames of sandboxed code."""
    frames = [
        frame for frame in traceback.extract_tb(exc.__traceback__)
        if frame.filename.startswith("<sandbox")
    ]
    lines: list[str] = []
    if frames:
        lines.append("Traceback (most recent call last):\n")
        lines.extend(traceback.format_list(frames))
    lines.extend(traceback.format_exception_only(type(exc), exc))
    return "".join(lines)


def _execute(
    code: str,
    namespace: dict[str, Any],
    filename: str,
    writer: _PipeWriter,
    final: dict[str, Any],
) -> Message:
    """Run one code entry and describe the outcome as a 'done' message.

    Like the Python REPL, the value of a trailing bare expression is
    printed when it is not None.
    """
    done: Message = {
        "type": "done", "error": None, "memory_error": False,
        "control": None, "exit_code": None,
    }
    try:
        tree = ast.parse(code, filename, "exec")
        security.validate_code(tree)
    except SyntaxError as exc:
        done["error"] = "".join(traceback.format_exception_only(
            type(exc), exc))
        return done
    except security.SecurityError as exc:
        done["error"] = f"SecurityError: {exc}"
        return done

    # Register the source so tracebacks can display the faulty lines.
    linecache.cache[filename] = (
        len(code), None, code.splitlines(keepends=True), filename)
    last_expr: ast.Expression | None = None
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        tree.body.pop()
        last_expr = ast.Expression(body=last.value)

    try:
        exec(compile(tree, filename, "exec"), namespace)
        if last_expr is not None:
            value = eval(compile(last_expr, filename, "eval"), namespace)
            if value is not None:
                writer.write(repr(value) + "\n")
    except _FinalAnswerSignal:
        pass
    except SystemExit as exc:
        done["control"] = "SystemExit"
        done["exit_code"] = exc.code if isinstance(
            exc.code, (int, str, type(None))) else str(exc.code)
    except KeyboardInterrupt:
        done["control"] = "KeyboardInterrupt"
    except MemoryError:
        done["memory_error"] = True
        done["error"] = "MemoryError: the memory limit was exceeded"
    except BaseException as exc:
        done["error"] = _format_exception(exc)
    done["final_called"] = final["called"]
    done["final_answer"] = final["answer"]
    return done


def _preload_modules(allowlist: list[str]) -> None:
    """Import allowed modules before the audit hook locks file access."""
    for entry in allowlist:
        name = entry[:-2] if entry.endswith(".*") else entry
        try:
            importlib.import_module(name)
        except Exception:
            continue


def _enter_workdir(directories: list[str]) -> None:
    """Use the first existing allowed directory as working directory."""
    for directory in directories:
        if os.path.isdir(directory):
            os.chdir(directory)
            return


def worker_main(
    conn: Connection, config: dict[str, Any], tools: list[dict[str, Any]]
) -> None:
    """Entry point of the worker process.

    Args:
        conn: Worker end of the pipe.
        config: SmithSandboxConfig dumped to a dict.
        tools: [{"name": python_name, "params": [param names]}, ...]
    """
    try:
        # Ctrl+C is handled by the parent, which restarts the worker.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Never leak secrets (API keys) to the sandboxed code.
        os.environ.clear()
        _preload_modules(config["authorized_imports"])
        _enter_workdir(config["allowed_directories"])
        security.apply_memory_limit(config["max_memory_mb"])
        writer = _PipeWriter(conn, config["max_output_chars"])
        sys.stdout = writer
        sys.stderr = writer
        sys.stdin = io.StringIO()
        policy = security.install_audit_policy(
            config["allowed_directories"])
        safe_builtins = security.build_safe_builtins(
            config["authorized_imports"])
    except Exception:
        send(conn, {"type": "fatal", "error": traceback.format_exc()})
        return

    final: dict[str, Any] = {"called": False, "answer": None}

    def final_answer(answer: object) -> None:
        """Submit the final answer and stop the current execution."""
        final["called"] = True
        final["answer"] = answer if isinstance(answer, str) else str(answer)
        raise _FinalAnswerSignal()

    injected: dict[str, Any] = {
        spec["name"]: _make_tool_stub(conn, spec["name"], spec["params"])
        for spec in tools
    }
    injected["final_answer"] = final_answer
    injected["ToolError"] = ToolError
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins, "__name__": "__sandbox__",
    }

    policy.activate()
    send(conn, {"type": "ready"})

    counter = 0
    while True:
        try:
            message = recv(conn)
        except (EOFError, OSError, ValueError):
            return
        if message.get("type") != "exec":
            return
        counter += 1
        writer.reset()
        final["called"] = False
        final["answer"] = None
        # Re-inject so overwriting a tool name does not break later steps.
        namespace.update(injected)
        done = _execute(
            message["code"], namespace, f"<sandbox-{counter}>",
            writer, final)
        done["truncated"] = writer.truncated
        send(conn, done)
