"""Minimal MCP server used by the sandbox tests.

Usage: python dummy_mcp_server.py [stdio | streamable-http PORT]
"""

import sys
import time

from mcp.server.fastmcp import FastMCP

transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
server = FastMCP("dummy", port=port)


@server.tool()
def add(a: int, b: int = 1) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@server.tool()
def big_output(size: int) -> str:
    """Return a string of `size` characters."""
    return "x" * size


@server.tool()
def slow(seconds: float) -> str:
    """Sleep, then answer."""
    time.sleep(seconds)
    return "slept"


@server.tool()
def fail(message: str) -> str:
    """Always fail with the given message."""
    raise ValueError(message)


@server.tool(name="weird-name")
def weird_name() -> str:
    """Tool whose name is not a valid Python identifier."""
    return "weird"


@server.resource("info://about")
def about() -> str:
    """Describe the dummy server."""
    return "dummy server for tests"


@server.prompt()
def greet(name: str) -> str:
    """Greeting prompt."""
    return f"Hello {name}"


if __name__ == "__main__":
    server.run(transport)  # type: ignore[arg-type]
