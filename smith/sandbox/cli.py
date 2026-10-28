"""Entry point for `uv run sandbox` (interactive REPL).

Usage:
    uv run sandbox [config.json] [--mcp-stdio "CMD" | --mcp-server URL]
"""

import argparse
import codeop
import sys

from smith.sandbox.config import SandboxConfigError, load_config
from smith.sandbox.mcp_client import McpClient, McpConnectionError
from smith.sandbox.sandbox import Sandbox, SandboxError

EXIT_COMMAND = "exit"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="sandbox",
        description="Interactive sandbox for LLM-generated Python code.")
    parser.add_argument(
        "config", nargs="?", help="sandbox configuration JSON file")
    server = parser.add_mutually_exclusive_group()
    server.add_argument(
        "--mcp-stdio", metavar="COMMAND",
        help="launch an MCP server and talk to it over stdio")
    server.add_argument(
        "--mcp-server", metavar="URL",
        help="connect to a streamable HTTP MCP server")
    parser.add_argument(
        "--manual", action="store_true",
        help="print the generated LLM manual at startup")
    return parser.parse_args(argv)


def read_entry() -> str | None:
    """Read one complete entry, prompting '...' for continuation lines.

    Returns:
        The source code, or None when the user typed 'exit'.

    Raises:
        EOFError: Ctrl+D was pressed.
    """
    lines: list[str] = []
    prompt = ">>> "
    while True:
        line = input(prompt)
        if not lines and line.strip() == EXIT_COMMAND:
            return None
        lines.append(line)
        source = "\n".join(lines)
        try:
            complete = codeop.compile_command(source, "<input>", "single")
        except (SyntaxError, OverflowError, ValueError):
            return source  # let the sandbox report the error
        if complete is not None:
            return source
        prompt = "... "


def run_repl(sandbox: Sandbox, show_manual: bool) -> None:
    """Read-eval-print loop until 'exit', Ctrl+D or SystemExit."""
    try:
        import readline  # noqa: F401  (line editing and history)
    except ImportError:
        pass
    tools = ", ".join(["final_answer", *sandbox.tool_names])
    print(f"Agent Smith sandbox. Available functions: {tools}")
    print(f"Type '{EXIT_COMMAND}' or press Ctrl+D to quit.")
    if show_manual:
        print(sandbox.manual)
    while True:
        try:
            source = read_entry()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            continue
        if source is None:
            return
        if not source.strip():
            continue
        try:
            result = sandbox.execute(source)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: execution stopped, sandbox "
                  "restarted (variables lost)")
            continue
        text = result.to_observation(empty_message="")
        if text:
            print(text)
        if result.final_answer_called:
            print(f"[final_answer] {result.final_answer}")


def main(argv: list[str] | None = None) -> None:
    """Start the interactive sandbox."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except SandboxConfigError as exc:
        sys.exit(f"sandbox: {exc}")
    mcp: McpClient | None = None
    if args.mcp_stdio or args.mcp_server:
        mcp = McpClient(
            "stdio" if args.mcp_stdio else "http",
            args.mcp_stdio or args.mcp_server,
            call_timeout=config.mcp_call_timeout_seconds)
    try:
        if mcp is not None:
            mcp.connect()
        with Sandbox(config, mcp) as sandbox:
            run_repl(sandbox, args.manual)
    except (McpConnectionError, SandboxError) as exc:
        sys.exit(f"sandbox: {exc}")
    finally:
        if mcp is not None:
            mcp.close()


if __name__ == "__main__":
    main()
