"""Synchronous MCP client (stdio or streamable HTTP).

The MCP SDK is asynchronous while the sandbox is synchronous. The
client therefore runs its own asyncio event loop in a background
thread; public methods submit coroutines to that loop and wait for
their result.
"""

import asyncio
import json
import keyword
import os
import re
import shlex
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import AnyUrl, BaseModel, Field


class McpConnectionError(Exception):
    """Raised when the MCP server cannot be reached or used."""


class ToolParam(BaseModel):
    """One parameter of an MCP tool, derived from its JSON schema."""

    name: str
    type_hint: str = "Any"
    required: bool = True
    default: Any = None
    description: str = ""


class ToolSpec(BaseModel):
    """An MCP tool as exposed in the sandbox."""

    name: str = Field(description="Name on the MCP server")
    python_name: str = Field(description="Callable name in the sandbox")
    description: str = ""
    params: list[ToolParam] = Field(default_factory=list)

    @property
    def param_names(self) -> list[str]:
        """Parameter names in positional order."""
        return [param.name for param in self.params]


JSON_TYPES = {
    "string": "str", "integer": "int", "number": "float",
    "boolean": "bool", "array": "list", "object": "dict", "null": "None",
}


def schema_to_type_hint(schema: dict[str, Any]) -> str:
    """Translate a JSON schema fragment into a Python type hint string."""
    for key in ("anyOf", "oneOf"):
        if key in schema:
            hints = [schema_to_type_hint(sub) for sub in schema[key]]
            return " | ".join(dict.fromkeys(hints))
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(JSON_TYPES.get(k, "Any") for k in kind)
    if kind == "array" and isinstance(schema.get("items"), dict):
        return f"list[{schema_to_type_hint(schema['items'])}]"
    return JSON_TYPES.get(kind, "Any") if isinstance(kind, str) else "Any"


def python_identifier(name: str) -> str:
    """Turn an MCP tool name into a valid, non-private Python name."""
    identifier = re.sub(r"\W", "_", name)
    if not identifier or not identifier[0].isalpha():
        identifier = f"tool_{identifier.lstrip('_')}"
    if keyword.iskeyword(identifier):
        identifier += "_"
    return identifier


def parse_tool(tool: types.Tool) -> ToolSpec:
    """Build a ToolSpec from an MCP tool definition.

    Required parameters come first so that the published signature is
    valid Python and positional calls map to the right names.
    """
    schema = tool.inputSchema or {}
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    params = [
        ToolParam(
            name=name,
            type_hint=schema_to_type_hint(prop),
            required=name in required,
            default=prop.get("default"),
            description=prop.get("description", "") or "",
        )
        for name, prop in properties.items()
    ]
    params.sort(key=lambda param: not param.required)
    return ToolSpec(
        name=tool.name,
        python_name=python_identifier(tool.name),
        description=tool.description or "",
        params=params,
    )


def _content_to_text(content: Any) -> str:
    """Render one MCP content block (text, image, resource...) as text."""
    text = getattr(content, "text", None)
    if text is None:
        resource = getattr(content, "resource", None)
        text = getattr(resource, "text", None)
    if text is None:
        return f"[{getattr(content, 'type', 'unknown')} content omitted]"
    return str(text)


class McpClient:
    """Connection to one MCP server with a blocking API."""

    def __init__(
        self,
        transport: Literal["stdio", "http"],
        target: str,
        call_timeout: float = 600.0,
        connect_timeout: float = 30.0,
    ) -> None:
        """Prepare (but do not open) the connection.

        Args:
            transport: 'stdio' (target is a shell command) or 'http'
                (target is the streamable HTTP endpoint URL).
            target: Command or URL.
            call_timeout: Max seconds for one request.
            connect_timeout: Max seconds to connect and discover tools.
        """
        self.transport = transport
        self.target = target
        self._call_timeout = call_timeout
        self._connect_timeout = connect_timeout
        self.tools: list[ToolSpec] = []
        self.has_resources = False
        self.has_prompts = False
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def __enter__(self) -> "McpClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def connect(self) -> None:
        """Open the connection and discover tools, resources and prompts.

        Raises:
            McpConnectionError: The server failed to start or answer.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._main()),
            name="mcp-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._connect_timeout):
            self.close()
            raise McpConnectionError(
                f"MCP server '{self.target}' did not answer within "
                f"{self._connect_timeout:.0f}s")
        if self._session is None:
            error = self._error
            self.close()
            raise McpConnectionError(
                f"cannot connect to MCP server '{self.target}': "
                f"{_describe(error)}")

    def close(self) -> None:
        """Close the session and stop the background thread."""
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass  # loop already closed
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._session = None

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                session = await self._open_session(stack)
                await self._discover(session)
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._error = exc
        finally:
            self._session = None
            self._ready.set()

    async def _open_session(self, stack: AsyncExitStack) -> ClientSession:
        if self.transport == "stdio":
            argv = shlex.split(self.target)
            if not argv:
                raise McpConnectionError("empty MCP stdio command")
            # The full environment is passed on purpose: SWE-bench tools
            # need variables such as TESTBED_PATH.
            params = StdioServerParameters(
                command=argv[0], args=argv[1:], env=dict(os.environ))
            read, write = await stack.enter_async_context(
                stdio_client(params))
        else:
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(self.target))
        return await stack.enter_async_context(ClientSession(read, write))

    async def _discover(self, session: ClientSession) -> None:
        capabilities = (await session.initialize()).capabilities
        if capabilities.tools is not None:
            result = await session.list_tools()
            tools = list(result.tools)
            while result.nextCursor:
                result = await session.list_tools(cursor=result.nextCursor)
                tools.extend(result.tools)
            self.tools = [parse_tool(tool) for tool in tools]
        self.has_resources = capabilities.resources is not None
        self.has_prompts = capabilities.prompts is not None

    def _request(self, method: str, *args: Any) -> Any:
        """Call a ClientSession coroutine method and wait for its result."""
        session, loop = self._session, self._loop
        if session is None or loop is None:
            raise McpConnectionError("MCP client is not connected")
        future = asyncio.run_coroutine_threadsafe(
            getattr(session, method)(*args), loop)
        try:
            return future.result(timeout=self._call_timeout)
        except FutureTimeoutError:
            future.cancel()
            raise McpConnectionError(
                f"MCP request timed out after {self._call_timeout:.0f}s")

    def call_tool(self, name: str, arguments: dict[str, Any]
                  ) -> tuple[bool, str]:
        """Call a tool. Returns (ok, text)."""
        result: types.CallToolResult = self._request(
            "call_tool", name, arguments)
        parts = [_content_to_text(block) for block in result.content]
        if not parts and result.structuredContent is not None:
            parts.append(json.dumps(result.structuredContent))
        return not result.isError, "\n".join(parts)

    def list_resources(self) -> str:
        """List the server resources, one per line."""
        result: types.ListResourcesResult = self._request("list_resources")
        lines = [
            f"{res.uri} - {res.name}"
            + (f": {res.description}" if res.description else "")
            for res in result.resources
        ]
        return "\n".join(lines) or "(no resources)"

    def read_resource(self, uri: str) -> str:
        """Read a resource and return its text content."""
        result: types.ReadResourceResult = self._request(
            "read_resource", AnyUrl(uri))
        return "\n".join(_content_to_text(item) for item in result.contents)

    def list_prompts(self) -> str:
        """List the server prompts with their arguments."""
        result: types.ListPromptsResult = self._request("list_prompts")
        lines = []
        for prompt in result.prompts:
            args = ", ".join(
                arg.name + ("" if arg.required else "?")
                for arg in prompt.arguments or [])
            line = f"{prompt.name}({args})"
            if prompt.description:
                line += f" - {prompt.description}"
            lines.append(line)
        return "\n".join(lines) or "(no prompts)"

    def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> str:
        """Render a prompt as 'role: text' lines."""
        result: types.GetPromptResult = self._request(
            "get_prompt", name, arguments)
        return "\n".join(
            f"{message.role}: {_content_to_text(message.content)}"
            for message in result.messages)


def _describe(error: BaseException | None) -> str:
    """Readable message, unwrapping anyio exception groups."""
    if error is None:
        return "unknown error"
    inner = getattr(error, "exceptions", None)
    if inner:
        return "; ".join(_describe(sub) for sub in inner)
    return f"{type(error).__name__}: {error}"
