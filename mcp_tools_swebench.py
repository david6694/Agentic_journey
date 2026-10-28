"""MCP server exposing the 9 mandatory tools for the SWE-bench benchmark."""
import ast
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import List

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("swebench-tools")

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}
MAX_OUTPUT_CHARS = 20000


def _testbed() -> Path:
    return Path(os.environ.get("TESTBED_PATH", "/testbed")).resolve()


def _resolve_path(filepath: str) -> Path:
    root = _testbed()
    p = Path(filepath)
    target = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if target != root and root not in target.parents:
        raise ValueError(
            f"Access denied: {filepath} is outside TESTBED_PATH ({root})"
        )
    return target


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n... [Output truncated to {MAX_OUTPUT_CHARS} characters]"
    )


def _iter_files(base: Path, file_pattern: str = "*") -> List[Path]:
    results: List[Path] = []
    for root_dir, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for fname in sorted(files):
            full = Path(root_dir) / fname
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
        matches = [str(p.resolve()) for p in _iter_files(base, pattern)]
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
                matches.append(f"{path.resolve()}:{idx} {line}")
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
                matches.append(f"{path.resolve()}:{idx} {line}")
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
                matches.append(f"{path.resolve()}:{idx} {content}")
    return _truncate("\n".join(matches)) if matches else "No references found."


@mcp.tool()
def run_tests() -> str:
    """Execute the evaluation script or test suite in TESTBED_PATH."""
    root = _testbed()
    eval_script = os.environ.get("EVAL_SCRIPT", "")
    eval_file = root / ".smith_eval.sh"
    if eval_script:
        cmd = ["bash", "-c", eval_script]
    elif eval_file.is_file():
        cmd = ["bash", str(eval_file)]
    else:
        cmd = ["pytest", "-q"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        return _truncate(out.strip())
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out after 120 seconds."
    except Exception as exc:
        return f"Error running tests: {exc}"


@mcp.tool()
def get_patch() -> str:
    """Retrieve the unified git diff of all changes made to the repository."""
    try:
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
    try:
        cwd = _resolve_path(workdir) if workdir else _testbed()
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        return _truncate(out.strip())
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as exc:
        return f"Error executing command: {exc}"


if __name__ == "__main__":
    mcp.run()
