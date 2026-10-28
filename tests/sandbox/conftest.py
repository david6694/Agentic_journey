"""Shared fixtures for the sandbox tests."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from smith.sandbox.config import SmithSandboxConfig
from smith.sandbox.mcp_client import McpClient
from smith.sandbox.sandbox import Sandbox

DUMMY_SERVER = Path(__file__).parent / "fixtures" / "dummy_mcp_server.py"


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """An allowed directory for the sandbox."""
    directory = tmp_path / "agent"
    directory.mkdir()
    return directory


@pytest.fixture
def config(workdir: Path) -> SmithSandboxConfig:
    """Fast limits to keep the tests short."""
    return SmithSandboxConfig(
        allowed_directories=[str(workdir)],
        max_execution_time_seconds=2,
        max_memory_mb=200,
        max_output_chars=1000,
        max_tool_output_chars=500,
    )


@pytest.fixture
def sandbox(config: SmithSandboxConfig) -> Iterator[Sandbox]:
    """A started sandbox without MCP server."""
    with Sandbox(config) as box:
        yield box


@pytest.fixture
def mcp_stdio() -> Iterator[McpClient]:
    """Client connected to the dummy server over stdio."""
    command = f'"{sys.executable}" "{DUMMY_SERVER}"'
    with McpClient("stdio", command, call_timeout=30) as client:
        yield client


@pytest.fixture
def mcp_sandbox(
    config: SmithSandboxConfig, mcp_stdio: McpClient
) -> Iterator[Sandbox]:
    """A sandbox exposing the dummy server tools."""
    with Sandbox(config, mcp_stdio) as box:
        yield box
