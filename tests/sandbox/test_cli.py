"""`uv run sandbox` fed by a pipe, as the exam scripts do."""

import subprocess
import sys

from smith.sandbox.cli import _split_script

SCRIPT = """x = 3
if x >= 1:
    print("if branch")

    # comment after a blank line
    print("after blank line")
elif x == 0:
    print("elif branch")
import os
print("after blocked import")
for i in range(1):
    print("last block")"""


def _pipe(script: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", "smith.sandbox.cli"], input=script,
        capture_output=True, text=True, timeout=60).stdout


def test_piped_script_runs_whole_blocks_and_survives_errors() -> None:
    output = _pipe(SCRIPT)
    assert "if branch" in output and "after blank line" in output
    assert "IndentationError" not in output and "SyntaxError" not in output
    assert "ImportError" in output and "after blocked import" in output
    assert "last block" in output


def test_piped_exit_stops_the_script() -> None:
    output = _pipe('print("before")\nexit\nprint("after")\n')
    assert "before" in output and "after" not in output


def test_split_keeps_decorators_and_shared_lines() -> None:
    source = "a = 1; b = 2\n@staticmethod\ndef f():\n\n    return 1\n"
    assert _split_script(source) == [
        "a = 1; b = 2", "@staticmethod\ndef f():\n\n    return 1"
    ]


def test_split_leaves_invalid_scripts_whole() -> None:
    assert _split_script("if True\n    pass") == ["if True\n    pass"]


def test_split_keeps_exit_inside_a_string() -> None:
    source = 't = """a\nexit\nb"""\nprint(t)\n'
    assert _split_script(source) == ['t = """a\nexit\nb"""', "print(t)"]


def test_piped_null_byte_is_reported_without_crashing() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "smith.sandbox.cli"], input="x = 1\0\n",
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "Traceback" not in result.stderr
