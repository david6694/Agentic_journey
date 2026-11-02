"""Tests for mcp_tools_swebench.py."""
import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp_tools_swebench as swe


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path),
        check=True,
    )
    sample = tmp_path / "calc.py"
    sample.write_text(
        "class Calculator:\n"
        "    pass\n\n"
        "def compute_sum(a, b):\n"
        "    return a - b\n\n"
        "def caller():\n"
        "    return compute_sum(2, 3)\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_calc.py"
    test_file.write_text(
        "from calc import compute_sum\n\n"
        "def test_sum():\n"
        "    assert compute_sum(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True
    )
    return sample


def test_swebench_tools_direct(tmp_path: Path, monkeypatch: object) -> None:
    sample = _init_git_repo(tmp_path)
    os.environ["TESTBED_PATH"] = str(tmp_path)

    # list_files
    files = swe.list_files(".", "*.py")
    assert str(sample.resolve()) in files

    # read_file (cat -n style)
    content = swe.read_file("calc.py", 1, 5)
    assert "1: class Calculator:" in content
    assert "4: def compute_sum(a, b):" in content

    # search_code
    search_res = swe.search_code("compute_sum", "*.py")
    assert f"{sample.resolve()}:4 def compute_sum(a, b):" in search_res

    # search_function_or_class_definition_in_code
    def_fn = swe.search_function_or_class_definition_in_code("compute_sum")
    assert f"{sample.resolve()}:4 def compute_sum(a, b):" in def_fn
    def_cls = swe.search_function_or_class_definition_in_code("Calculator")
    assert f"{sample.resolve()}:1 class Calculator:" in def_cls

    # find_references
    refs = swe.find_references("compute_sum")
    assert f"{sample.resolve()}:8     return compute_sum(2, 3)" in refs

    # edit_file syntax check
    bad_edit = swe.edit_file("calc.py", "return a - b", "return a +")
    assert "SyntaxError" in bad_edit

    # edit_file valid
    ok_edit = swe.edit_file("calc.py", "return a - b", "return a + b")
    assert "Successfully updated" in ok_edit

    # get_patch
    patch = swe.get_patch()
    assert "-    return a - b" in patch
    assert "+    return a + b" in patch

    # run_tests
    os.environ["EVAL_SCRIPT"] = f"{shlex.quote(sys.executable)} -m pytest -q"
    test_out = swe.run_tests()
    assert "exit_code: 0" in test_out

    # run_command
    cmd_out = swe.run_command("echo hello_swe")
    assert "exit_code: 0" in cmd_out
    assert "hello_swe" in cmd_out


def test_tools_do_not_follow_symlinks_out_of_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.txt").write_text("INSIDE\n", encoding="utf-8")
    (repo / "link.txt").symlink_to(repo / "inside.txt")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET\n", encoding="utf-8")
    (repo / "leak.txt").symlink_to(secret)
    monkeypatch.setenv("TESTBED_PATH", str(repo))
    monkeypatch.delenv("SWE_CONTAINER_ID", raising=False)
    assert swe.search_code("TOP_SECRET") == "No matches found."
    assert swe.find_references("TOP_SECRET") == "No references found."
    listed = swe.list_files()
    assert "leak.txt" not in listed and "link.txt" in listed
    assert "INSIDE" in swe.search_code("INSIDE")


def test_swebench_mcp_stdio_handshake(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    async def _check() -> None:
        root = Path(__file__).resolve().parent.parent
        server_script = str(root / "mcp_tools_swebench.py")
        env = dict(os.environ)
        env["TESTBED_PATH"] = str(tmp_path)
        params = StdioServerParameters(
            command=sys.executable, args=[server_script], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                expected = {
                    "read_file",
                    "edit_file",
                    "list_files",
                    "search_code",
                    "search_function_or_class_definition_in_code",
                    "find_references",
                    "run_tests",
                    "get_patch",
                    "run_command",
                }
                assert expected.issubset(names)
                res = await session.call_tool(
                    "read_file",
                    {"filepath": "calc.py", "start_line": 1, "end_line": 2},
                )
                text = res.content[0].text  # type: ignore[union-attr]
                assert "1: class Calculator:" in text

    asyncio.run(_check())
