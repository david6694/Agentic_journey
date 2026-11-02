"""Tests for mcp_tools_mbpp.py."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_tools_mbpp import run_tests


def test_run_tests_direct() -> None:
    code_ok = "def add(a, b):\n    return a + b\n"
    res = json.loads(run_tests(code_ok, ["assert add(1, 2) == 3"]))
    assert res["success"] is True
    assert "All 1 tests passed" in res["output"]

    code_ko = "def add(a, b):\n    return a - b\n"
    fail = json.loads(run_tests(code_ko, ["assert add(1, 2) == 3"]))
    assert fail["success"] is False
    assert "Failed test 1" in fail["output"]


def test_run_tests_provides_math_like_the_moulinette() -> None:
    code = "def half(x):\n    return x / 2\n"
    res = json.loads(run_tests(code, ["assert math.isclose(half(3), 1.5)"]))
    assert res["success"] is True


def test_mbpp_mcp_stdio_handshake() -> None:
    async def _check() -> None:
        root = Path(__file__).resolve().parent.parent
        server_script = str(root / "mcp_tools_mbpp.py")
        params = StdioServerParameters(
            command=sys.executable, args=[server_script]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                assert "run_tests" in names
                call_res = await session.call_tool(
                    "run_tests",
                    {
                        "code": "def mul(x, y): return x * y",
                        "test_list": ["assert mul(3, 4) == 12"],
                    },
                )
                text = call_res.content[0].text  # type: ignore[union-attr]
                parsed = json.loads(text)
                assert parsed["success"] is True

    asyncio.run(_check())
