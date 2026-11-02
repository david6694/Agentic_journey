"""Sandbox manual: the part of the system prompt describing the sandbox.

It is generated from the connected MCP server's tool schemas, so it
automatically follows whatever server is plugged in.
"""

from collections.abc import Sequence

from smith.sandbox.config import SmithSandboxConfig
from smith.sandbox.mcp_client import ToolParam, ToolSpec

RESOURCE_HELPERS = (
    ("list_mcp_resources() -> str", "List the server resources."),
    ("read_mcp_resource(uri: str) -> str", "Read one resource."),
)
TEMPLATE_HELPERS = (
    ("list_mcp_prompts() -> str", "List the server prompt templates."),
    ("get_mcp_prompt(name: str, arguments: dict | None = None) -> str",
     "Render one prompt template."),
)


def format_param(param: ToolParam) -> str:
    """Render one parameter as it would appear in a Python signature."""
    text = f"{param.name}: {param.type_hint}"
    if not param.required:
        text += f" = {param.default!r}"
    return text


def format_signature(tool: ToolSpec) -> str:
    """Render a tool as a Python signature, e.g. `f(a: int) -> str`."""
    params = ", ".join(format_param(param) for param in tool.params)
    return f"{tool.python_name}({params}) -> str"


def _first_paragraph(text: str) -> str:
    """Keep the first paragraph of a description to save tokens."""
    return text.strip().split("\n\n")[0].strip()


def _format_tool(tool: ToolSpec) -> list[str]:
    lines = [format_signature(tool)]
    if tool.description:
        for line in _first_paragraph(tool.description).splitlines():
            lines.append(f"    {line.strip()}")
    for param in tool.params:
        if param.description:
            lines.append(f"    - {param.name}: {param.description}")
    return lines


def build_manual(
    config: SmithSandboxConfig,
    tools: Sequence[ToolSpec] = (),
    has_resources: bool = False,
    has_prompts: bool = False,
) -> str:
    """Generate the sandbox manual fed to the LLM.

    Args:
        config: Active sandbox limits (imports, directories...).
        tools: Tools discovered on the MCP server.
        has_resources: Expose the resource helpers.
        has_prompts: Expose the prompt helpers.

    Returns:
        The manual as plain text.
    """
    imports = ", ".join(config.authorized_imports) or "none"
    directories = ", ".join(config.allowed_directories) or "none"
    lines = [
        "# Sandbox",
        "Your code runs in a Python 3.10 sandbox.",
        "- Variables persist between executions, unless you are told the "
        "sandbox was restarted.",
        "- You only see what you print(), plus the value of a final bare "
        "expression.",
        f"- Allowed imports: {imports}.",
        f"- File access limited to: {directories}.",
        "- No network, no subprocess. Names and attributes starting "
        "with '_' are forbidden.",
        f"- Limits per execution: {config.max_execution_time_seconds}s, "
        f"{config.max_memory_mb} MB, {config.max_output_chars} printed "
        "characters.",
        "- Tools are Python functions returning str. A failing tool "
        "raises ToolError.",
        "",
        "# Functions",
        "final_answer(answer: str) -> None",
        "    Submit your final answer. This ends the task.",
    ]
    for tool in tools:
        lines.extend(_format_tool(tool))
    if has_resources:
        for signature, description in RESOURCE_HELPERS:
            lines.extend([signature, f"    {description}"])
    if has_prompts:
        for signature, description in TEMPLATE_HELPERS:
            lines.extend([signature, f"    {description}"])
    return "\n".join(lines)
