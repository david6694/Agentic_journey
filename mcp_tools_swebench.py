"""MCP server exposing the 9 mandatory tools for the SWE-bench benchmark."""
import ast
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("swebench-tools")

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}
MAX_OUTPUT_CHARS = 20000
RUN_TESTS_TIMEOUT = 120
RUN_COMMAND_TIMEOUT = 60


def _testbed() -> Path:
    return Path(os.environ.get("TESTBED_PATH", "/testbed")).resolve()


def _container_id() -> str:
    return os.environ.get("SWE_CONTAINER_ID", "").strip()


def _resolve_path(filepath: str) -> Path:
    root = _testbed()
    p = Path(filepath)
    if p.is_absolute():
        try:
            rel_to_docker = p.relative_to("/testbed")
            target = (root / rel_to_docker).resolve()
        except ValueError:
            target = p.resolve()
    else:
        target = (root / p).resolve()
    if target != root and root not in target.parents:
        raise ValueError(
            f"Access denied: {filepath} is outside TESTBED_PATH ({root})"
        )
    return target


def _shown_path(path: Path) -> str:
    """Path to show the LLM: /testbed/... when a container is used."""
    if _container_id():
        return f"/testbed/{path.relative_to(_testbed()).as_posix()}"
    return str(path)


def _sync_to_container(target: Path) -> None:
    cid = _container_id()
    if not cid:
        return
    rel = target.relative_to(_testbed())
    subprocess.run(
        ["docker", "cp", str(target), f"{cid}:/testbed/{rel.as_posix()}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def _sync_from_container() -> None:
    # shortcut: files deleted in the container stay in the local copy, rare
    # for a bug fix; compare with `git status` if an agent trips over it.
    subprocess.run(
        ["docker", "cp", f"{_container_id()}:/testbed/.", str(_testbed())],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )


def _exec_in_container(
    argv: List[str], workdir: str, seconds: int
) -> "subprocess.CompletedProcess[str]":
    """Run a command in the container, then refresh the local copy.

    Killing `docker exec` leaves the command running in the container, so
    `timeout` stops it in there. The local limit is only a safety net.
    """
    proc = subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            workdir,
            _container_id(),
            "timeout",
            "-k",
            "5",
            str(seconds),
            *argv,
        ],
        capture_output=True,
        text=True,
        timeout=seconds + 10,
    )
    _sync_from_container()
    if proc.returncode == 124:
        raise subprocess.TimeoutExpired(argv, seconds)
    return proc


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n... [Output truncated to {MAX_OUTPUT_CHARS} characters]"
    )


def _iter_files(base: Path, file_pattern: str = "*") -> List[Path]:
    results: List[Path] = []
    root = _testbed()
    for root_dir, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for fname in sorted(files):
            full = Path(root_dir) / fname
            # A symlink must not expose files outside the repository.
            if not full.is_file() or root not in full.resolve().parents:
                continue
            rel = str(full.relative_to(base))
            if fnmatch.fnmatch(fname, file_pattern) or fnmatch.fnmatch(
                rel, file_pattern
            ):
                results.append(full)
    return results


@mcp.tool()
def read_file(filepath: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read the content of a file with line numbers (`<line>: <content>`)."""
    try:
        target = _resolve_path(filepath)
        if not target.is_file():
            return f"Error: File not found: {filepath}"
        lines = target.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        start = max(1, start_line)
        end = len(lines) if end_line < 1 else min(len(lines), end_line)
        out = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
        return _truncate("\n".join(out))
    except Exception as exc:
        return f"Error reading {filepath}: {exc}"


@mcp.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """Replace an exact string in a file with a new string."""
    try:
        target = _resolve_path(filepath)
        if not target.is_file():
            return f"Error: File not found: {filepath}"
        content = target.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {filepath}."
        if count > 1:
            return (
                f"Error: old_str appears {count} times in {filepath}. "
                "Include more context lines to make it unique."
            )
        updated = content.replace(old_str, new_str, 1)
        if target.suffix == ".py":
            try:
                ast.parse(updated, filename=str(target))
            except SyntaxError as syn:
                return (
                    "Error: Edit rejected because it introduces a "
                    f"SyntaxError at line {syn.lineno}: {syn.msg}"
                )
        target.write_text(updated, encoding="utf-8")
        _sync_to_container(target)
        return f"Successfully updated {filepath}."
    except Exception as exc:
        return f"Error editing {filepath}: {exc}"


@mcp.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory matching a given pattern."""
    try:
        base = _resolve_path(directory)
        if not base.is_dir():
            return f"Error: Directory not found: {directory}"
        matches = [_shown_path(p) for p in _iter_files(base, pattern)]
        return _truncate("\n".join(matches)) if matches else "No files found."
    except Exception as exc:
        return f"Error listing files in {directory}: {exc}"


@mcp.tool()
def search_code(pattern: str, file_pattern: str = "*") -> str:
    """Perform a grep-like search in the codebase.

    Format: `/absolute/path_to_file.py:<line_number> <line_content>`
    """
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    matches: List[str] = []
    for path in _iter_files(_testbed(), file_pattern):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{_shown_path(path)}:{idx} {line}")
    return _truncate("\n".join(matches)) if matches else "No matches found."


@mcp.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """Find the definition of a function or a class in the codebase.

    Format: `/absolute/path_to_file.py:<line_number> <line_content>`
    """
    matches: List[str] = []
    fallback_re = re.compile(
        rf"^\s*(?:async\s+def|def|class)\s+{re.escape(name)}\b"
    )
    for path in _iter_files(_testbed(), "*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        found_lines = set()
        try:
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) and node.name == name:
                    found_lines.add(node.lineno)
        except SyntaxError:
            pass
        for idx, line in enumerate(lines, 1):
            if idx in found_lines or fallback_re.search(line):
                matches.append(f"{_shown_path(path)}:{idx} {line}")
    return _truncate("\n".join(matches)) if matches else "No definition found."


@mcp.tool()
def find_references(name: str, filepath: str = "", line: int = 0) -> str:
    """Find all usages of a symbol (function or class).

    Format: `/absolute/path_to_file.py:<line_number> <line_content>`
    """
    sym_re = re.compile(rf"\b{re.escape(name)}\b")
    files = (
        [_resolve_path(filepath)]
        if filepath
        else _iter_files(_testbed(), "*.py")
    )
    matches: List[str] = []
    for path in files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, content in enumerate(text.splitlines(), 1):
            if sym_re.search(content):
                matches.append(f"{_shown_path(path)}:{idx} {content}")
    return _truncate("\n".join(matches)) if matches else "No references found."


@mcp.tool()
def run_tests() -> str:
    """Execute the evaluation script or test suite in TESTBED_PATH."""
    root = _testbed()
    cid = _container_id()
    eval_script = os.environ.get("EVAL_SCRIPT", "")
    eval_file = root / ".smith_eval.sh"
    try:
        if cid:
            script = eval_script or (
                eval_file.read_text(encoding="utf-8")
                if eval_file.is_file()
                else "pytest -q"
            )
            proc = _exec_in_container(
                ["bash", "-c", script], "/testbed", RUN_TESTS_TIMEOUT
            )
        else:
            if eval_script:
                cmd = ["bash", "-c", eval_script]
            elif eval_file.is_file():
                cmd = ["bash", str(eval_file)]
            else:
                cmd = ["pytest", "-q"]
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=RUN_TESTS_TIMEOUT,
            )
        out = (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        return _truncate(out.strip())
    except subprocess.TimeoutExpired:
        return (
            f"Error: Test execution timed out after {RUN_TESTS_TIMEOUT} "
            "seconds."
        )
    except Exception as exc:
        return f"Error running tests: {exc}"


@mcp.tool()
def get_patch() -> str:
    """Retrieve the unified git diff of all changes made to the repository."""
    cid = _container_id()
    try:
        if cid:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    cid,
                    "git",
                    "-c",
                    "core.fileMode=false",
                    "diff",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        else:
            proc = subprocess.run(
                ["git", "-c", "core.fileMode=false", "diff"],
                cwd=str(_testbed()),
                capture_output=True,
                text=True,
                timeout=30,
            )
        if proc.returncode != 0:
            return f"Error running git diff: {proc.stderr.strip()}"
        return proc.stdout
    except Exception as exc:
        return f"Error getting patch: {exc}"


@mcp.tool()
def run_command(command: str, workdir: str = "") -> str:
    """Execute a shell command in the specified working directory.

    Returns the command's stdout, stderr, and exit code.
    """
    cid = _container_id()
    try:
        cwd = _resolve_path(workdir) if workdir else _testbed()
        if cid:
            rel_cwd = cwd.relative_to(_testbed())
            container_cwd = (
                "/testbed"
                if str(rel_cwd) == "."
                else f"/testbed/{rel_cwd.as_posix()}"
            )
            proc = _exec_in_container(
                ["bash", "-c", command], container_cwd, RUN_COMMAND_TIMEOUT
            )
        else:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=RUN_COMMAND_TIMEOUT,
            )
        out = (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        return _truncate(out.strip())
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {RUN_COMMAND_TIMEOUT} seconds."
    except Exception as exc:
        return f"Error executing command: {exc}"


if __name__ == "__main__":
    mcp.run()
