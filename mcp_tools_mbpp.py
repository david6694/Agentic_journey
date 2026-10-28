"""MCP server exposing tools for the MBPP benchmark."""
import json
import subprocess
import sys
from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mbpp-tools")


@mcp.tool()
def run_tests(code: str, test_list: List[str]) -> str:
    """Run a candidate Python solution against a list of test assertions.

    Returns a JSON string with a 'success' boolean and an 'output' message.
    """
    runner_script = (
        "import json, sys, traceback\n"
        "payload = json.loads(sys.stdin.read())\n"
        "ns = {}\n"
        "try:\n"
        "    exec(payload['code'], ns)\n"
        "except Exception:\n"
        "    print(traceback.format_exc(), file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "for i, test in enumerate(payload['test_list'], 1):\n"
        "    try:\n"
        "        exec(test, ns)\n"
        "    except Exception:\n"
        "        print(f'Failed test {i}: {test}', file=sys.stderr)\n"
        "        print(traceback.format_exc(), file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "print(f'All {len(payload[\"test_list\"])} tests passed.')\n"
    )
    payload = json.dumps({"code": code, "test_list": test_list})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner_script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (proc.stdout + proc.stderr).strip()
        return json.dumps({"success": proc.returncode == 0, "output": output})
    except subprocess.TimeoutExpired as exc:
        out_str = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err_str = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        partial = (out_str + err_str).strip()
        msg = "Execution timed out after 10 seconds."
        if partial:
            msg = f"{msg}\nPartial output:\n{partial}"
        return json.dumps({"success": False, "output": msg})


if __name__ == "__main__":
    mcp.run()
