"""Import, builtin, network and filesystem restrictions."""

import subprocess
import sys
from pathlib import Path

import pytest

from smith.sandbox.sandbox import Sandbox
from smith.sandbox.security import is_import_allowed


@pytest.mark.parametrize("name, allowed", [
    ("math", True),
    ("collections.abc", True),
    ("itertools", True),
    ("itertools.chain", False),
    ("os", False),
    ("mathx", False),
])
def test_import_allowlist_rules(name: str, allowed: bool) -> None:
    allowlist = ["math", "collections.*", "itertools"]
    assert is_import_allowed(name, allowlist) is allowed


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from os import path",
    "import importlib",
    "from . import x",
])
def test_import_block(sandbox: Sandbox, code: str) -> None:
    result = sandbox.execute(code)
    assert result.error is not None and "ImportError" in result.error


def test_allowed_imports(sandbox: Sandbox) -> None:
    result = sandbox.execute(
        "import math, collections.abc\nfrom itertools import chain\n"
        "print(math.sqrt(16), list(chain([1], [2])))")
    assert result.error is None
    assert result.output == "4.0 [1, 2]\n"


@pytest.mark.parametrize("code", [
    "eval('1')",
    "exec('x = 1')",
    "compile('1', 'f', 'eval')",
    "globals()",
    "vars()",
    "input()",
    "breakpoint()",
])
def test_builtin_block(sandbox: Sandbox, code: str) -> None:
    result = sandbox.execute(code)
    assert result.error is not None and "NameError" in result.error


@pytest.mark.parametrize("code", [
    "().__class__.__bases__[0].__subclasses__()",
    "__import__('os')",
    "__builtins__",
    "import random\nrandom._os",
    "import typing\ntyping.sys",
    "def g():\n    yield 1\ng().gi_frame",
    "from random import _os",
])
def test_introspection_escape_block(sandbox: Sandbox, code: str) -> None:
    result = sandbox.execute(code)
    assert result.error is not None and "SecurityError" in result.error


def test_getattr_cannot_bypass_ast_check(sandbox: Sandbox) -> None:
    result = sandbox.execute("getattr((), '__class__')")
    assert result.error is not None and "AttributeError" in result.error


def test_network_block(sandbox: Sandbox) -> None:
    result = sandbox.execute("import socket")
    assert result.error is not None and "ImportError" in result.error


def test_getattr_forbidden_module_name(sandbox: Sandbox) -> None:
    result = sandbox.execute("import typing\ngetattr(typing, 'sy' + 's')")
    assert result.error is not None and "AttributeError" in result.error


AUDIT_SCRIPT = """
import os, socket, subprocess, sys
from smith.sandbox.security import install_audit_policy
install_audit_policy([{allowed!r}]).activate()
checks = {{
    "socket": lambda: socket.socket(),
    "subprocess": lambda: subprocess.run(["true"]),
    "system": lambda: os.system("true"),
    "read": lambda: open("/etc/hosts").read(),
    "listdir": lambda: os.listdir("/"),
}}
for name, check in checks.items():
    try:
        check()
        print(name, "ALLOWED")
    except PermissionError:
        print(name, "blocked")
"""


def test_audit_hook_last_line_of_defence(workdir: Path) -> None:
    """Operations reached without the restricted namespace are refused.

    Runs in a separate interpreter: an audit hook cannot be removed.
    """
    completed = subprocess.run(
        [sys.executable, "-c", AUDIT_SCRIPT.format(allowed=str(workdir))],
        capture_output=True, text=True, timeout=30, check=False)
    assert "ALLOWED" not in completed.stdout, completed.stdout
    assert completed.stdout.count("blocked") == 5, completed.stderr


def test_path_restrict_read_outside(sandbox: Sandbox) -> None:
    result = sandbox.execute("open('/etc/hosts').read()")
    assert result.error is not None and "PermissionError" in result.error


def test_path_restrict_traversal(sandbox: Sandbox, workdir: Path) -> None:
    result = sandbox.execute(f"open('{workdir}/../../secret', 'w')")
    assert result.error is not None and "PermissionError" in result.error


def test_path_allowed_directory(sandbox: Sandbox, workdir: Path) -> None:
    result = sandbox.execute(
        f"with open('{workdir}/a.txt', 'w') as f:\n    f.write('hi')\n"
        f"print(open('{workdir}/a.txt').read())")
    assert result.error is None
    assert result.output == "hi\n"
    assert (workdir / "a.txt").read_text() == "hi"


def test_relative_path_in_workdir(sandbox: Sandbox, workdir: Path) -> None:
    result = sandbox.execute("open('rel.txt', 'w').write('x')")
    assert result.error is None
    assert (workdir / "rel.txt").exists()


def test_pipe_cannot_be_hijacked(sandbox: Sandbox, tmp_path: Path) -> None:
    """Regression: forged bytes on the pipe must not run code in the parent.

    The parent used to unpickle worker messages: writing a malicious
    pickle frame on the pipe descriptor executed it outside the sandbox.
    """
    marker = tmp_path / "pwned"
    payload = f"cposix\nsystem\n(S'touch {marker}'\ntR."
    code = (
        f"payload = {payload!r}.encode()\n"
        "frame = len(payload).to_bytes(4, 'big') + payload\n"
        "for fd in range(3, 30):\n"
        "    try:\n"
        "        open(fd, 'wb', buffering=0, closefd=False).write(frame)\n"
        "    except Exception:\n"
        "        pass\n"
    )
    sandbox.execute(code)
    sandbox.execute("print(1)")
    assert not marker.exists()
