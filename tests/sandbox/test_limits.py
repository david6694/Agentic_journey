"""Timeout, memory, output limits and execution semantics."""

import pytest

from smith.sandbox.sandbox import Sandbox


def test_variables_persist(sandbox: Sandbox) -> None:
    sandbox.execute("x = 41")
    result = sandbox.execute("x + 1")
    assert result.output == "42\n"


def test_timeout_kills_and_restarts(sandbox: Sandbox) -> None:
    sandbox.execute("x = 1")
    result = sandbox.execute("print('start', flush=True)\nwhile True:\n"
                             "    pass")
    assert result.timed_out
    assert result.worker_restarted
    assert result.output == "start\n"  # partial output is kept
    assert "partial" in result.to_observation()
    after = sandbox.execute("print(x)")
    assert after.error is not None and "NameError" in after.error


def test_memory_limit(sandbox: Sandbox) -> None:
    result = sandbox.execute(
        "blocks = []\nwhile True:\n    blocks.append(b'a' * 10_000_000)")
    assert result.memory_exceeded
    assert sandbox.execute("print(1)").output == "1\n"


def test_output_truncated(sandbox: Sandbox) -> None:
    result = sandbox.execute("print('a' * 5000)")
    assert result.output_truncated
    assert len(result.output) == 1000
    assert "truncated" in result.to_observation()


def test_final_answer_stops_execution(sandbox: Sandbox) -> None:
    result = sandbox.execute(
        "final_answer('def f(): pass')\nprint('never printed')")
    assert result.final_answer_called
    assert result.final_answer == "def f(): pass"
    assert result.output == ""


def test_final_answer_not_swallowed(sandbox: Sandbox) -> None:
    result = sandbox.execute(
        "try:\n    final_answer(42)\nexcept Exception:\n    pass")
    assert result.final_answer_called
    assert result.final_answer == "42"


def test_error_traceback_points_to_line(sandbox: Sandbox) -> None:
    result = sandbox.execute("a = 1\nb = a / 0")
    assert result.error is not None
    assert "ZeroDivisionError" in result.error
    assert "line 2" in result.error


def test_syntax_error_reported(sandbox: Sandbox) -> None:
    result = sandbox.execute("def f(:")
    assert result.error is not None and "SyntaxError" in result.error


def test_system_exit_propagates(sandbox: Sandbox) -> None:
    with pytest.raises(SystemExit):
        sandbox.execute("raise SystemExit(3)")


def test_keyboard_interrupt_propagates(sandbox: Sandbox) -> None:
    with pytest.raises(KeyboardInterrupt):
        sandbox.execute("raise KeyboardInterrupt")


def test_no_output_message(sandbox: Sandbox) -> None:
    assert "print()" in sandbox.execute("y = 2").to_observation()
