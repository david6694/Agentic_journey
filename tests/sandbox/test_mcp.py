"""MCP discovery, tool calls through the sandbox and manual."""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

from smith.sandbox.config import SmithSandboxConfig
from smith.sandbox.mcp_client import McpClient, McpConnectionError
from smith.sandbox.sandbox import Sandbox

from .conftest import DUMMY_SERVER


def test_discovery(mcp_stdio: McpClient) -> None:
    names = {tool.python_name for tool in mcp_stdio.tools}
    assert {"add", "echo", "big_output", "fail", "weird_name"} <= names
    assert mcp_stdio.has_resources and mcp_stdio.has_prompts


def test_tool_call_positional_and_keyword(mcp_sandbox: Sandbox) -> None:
    result = mcp_sandbox.execute("print(add(2, b=3), add(4))")
    assert result.error is None
    assert result.output == "5 5\n"


def test_tool_error_raises_tool_error(mcp_sandbox: Sandbox) -> None:
    result = mcp_sandbox.execute(
        "try:\n    fail('boom')\nexcept ToolError as e:\n    print('caught')")
    assert result.output == "caught\n"


def test_tool_output_truncated(mcp_sandbox: Sandbox) -> None:
    result = mcp_sandbox.execute("r = big_output(2000)\nprint(len(r) > 0)")
    assert result.notes and "truncated" in result.notes[0]


def test_resources_and_prompts(mcp_sandbox: Sandbox) -> None:
    result = mcp_sandbox.execute(
        "print(list_mcp_resources())\n"
        "print(read_mcp_resource('info://about'))\n"
        "print(get_mcp_prompt('greet', {'name': 'Neo'}))")
    assert result.error is None
    assert "dummy server for tests" in result.output
    assert "Hello Neo" in result.output


def test_tool_time_not_counted_in_timeout(mcp_sandbox: Sandbox) -> None:
    """The sandbox timeout is 2s; the tool alone takes 3s."""
    result = mcp_sandbox.execute("print(slow(3))")
    assert not result.timed_out
    assert result.output == "slept\n"


def test_manual_lists_tools(mcp_sandbox: Sandbox) -> None:
    manual = mcp_sandbox.manual
    assert "add(a: int, b: int = 1) -> str" in manual
    assert "weird_name() -> str" in manual
    assert "final_answer(answer: str)" in manual
    assert "read_mcp_resource" in manual


def test_unknown_command_fails_cleanly() -> None:
    with pytest.raises(McpConnectionError):
        McpClient("stdio", "this-command-does-not-exist",
                  connect_timeout=10).connect()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def http_url() -> Iterator[str]:
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, str(DUMMY_SERVER), "streamable-http", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
    server.terminate()
    server.wait(timeout=5)


def test_http_transport(http_url: str, config: SmithSandboxConfig) -> None:
    with McpClient("http", http_url, call_timeout=30) as client:
        with Sandbox(config, client) as box:
            result = box.execute("print(echo('over http'))")
    assert result.output == "over http\n"
