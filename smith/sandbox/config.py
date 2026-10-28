"""Sandbox configuration: Pydantic model and JSON loading."""

import json
from pathlib import Path

from pydantic import Field, ValidationError

from smith.models import SandboxConfig

DEFAULT_AUTHORIZED_IMPORTS = (
    "math", "math.*", "collections", "collections.*", "itertools", "re",
    "json", "typing", "typing.*", "functools", "operator", "heapq",
    "bisect", "copy", "string", "random", "datetime", "datetime.*",
    "array", "cmath",
)


class SandboxConfigError(Exception):
    """Raised when a sandbox configuration file cannot be loaded."""


class SmithSandboxConfig(SandboxConfig):
    """Moulinette's SandboxConfig extended with our own limits.

    The parent fields (imports, directories, time, memory) are kept
    untouched so any moulinette-compatible JSON file stays valid.
    models_public.py defaults both lists to [], so the subject's
    defaults are restored here.
    """

    authorized_imports: list[str] = Field(
        default_factory=lambda: list(DEFAULT_AUTHORIZED_IMPORTS))
    allowed_directories: list[str] = Field(
        default_factory=lambda: ["/testbed", "/tmp/agent"])
    max_output_chars: int = Field(
        default=10_000,
        description="Max characters of stdout kept per execution",
    )
    max_tool_output_chars: int = Field(
        default=20_000,
        description="Max characters returned by a single tool call",
    )
    mcp_call_timeout_seconds: float = Field(
        default=600.0,
        description="Max duration of one MCP request (not sandbox time)",
    )


def load_config(path: str | Path | None = None) -> SmithSandboxConfig:
    """Load a sandbox configuration from a JSON file.

    Args:
        path: JSON file to load. None returns the default configuration.

    Returns:
        The validated configuration.

    Raises:
        SandboxConfigError: The file is missing, is not JSON or does not
            match the schema.
    """
    if path is None:
        return SmithSandboxConfig()
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SandboxConfigError(f"cannot read config '{path}': {exc}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SandboxConfigError(f"invalid JSON in '{path}': {exc}")
    try:
        return SmithSandboxConfig.model_validate(data)
    except ValidationError as exc:
        raise SandboxConfigError(f"invalid config in '{path}':\n{exc}")
