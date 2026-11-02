"""Extract executable Python code from an LLM response.

Python blocks are the expected format. Models trained on other tool
calling styles (XML, JSON, ReAct) are converted to equivalent Python
calls, so the sandbox only ever receives Python.
"""

import json
import re
from typing import Any

NO_CODE_NOTE = (
    "No code block found. Answer with a Thought, then "
    "```python\n...\n```<end_code>"
)

# Any fenced block is matched, so the closing fence of a ```json block is
# never mistaken for an opening one; only Python tags are kept. A block closes
# on a fence alone on its line or ending the text: code may contain fences,
# such as a final_answer() string quoting a Markdown block.
FENCED_BLOCK = re.compile(
    r"```([\w+-]*)[ \t]*\n(.*?)"
    r"(^```[ \t]*(?:<end_code>)?[ \t]*$|```(?:<end_code>)?\s*\Z|\Z)",
    re.DOTALL | re.MULTILINE)
PYTHON_TAGS = ("", "python", "py", "python3")
XML_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.DOTALL)
XML_PARAMETER = re.compile(
    r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)
JSON_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
REACT_ACTION = re.compile(r"Action:\s*(\S+)\s*\nAction Input:\s*(.*)")


def _python_call(name: str, args: dict[str, Any]) -> str | None:
    """Render a tool call as Python, or None if names are not valid.

    Values go through repr(), so their content can never become code.
    """
    if not name.isidentifier() or not all(k.isidentifier() for k in args):
        return None
    params = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"result = {name}({params})\nprint(result)"


def _from_python_blocks(text: str) -> tuple[str | None, str | None]:
    blocks = [(body, end) for tag, body, end in FENCED_BLOCK.findall(text)
              if tag.lower() in PYTHON_TAGS]
    code = "\n\n".join(body.strip() for body, _ in blocks if body.strip())
    if not code:
        return None, None
    unclosed = blocks[-1][1] == ""
    note = ("The code block was not closed with ```; it was run up to the "
            "end of your answer.") if unclosed else None
    return code, note


def _from_xml(text: str) -> list[str | None]:
    return [
        _python_call(name, dict(XML_PARAMETER.findall(body)))
        for name, body in XML_INVOKE.findall(text)
    ]


def _from_json(text: str) -> list[str | None]:
    calls: list[str | None] = []
    for raw in JSON_TOOL_CALL.findall(text):
        try:
            call = json.loads(raw)
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(_python_call(call["name"], args))
        except (ValueError, KeyError, TypeError, AttributeError):
            calls.append(None)
    return calls


def _from_react(text: str) -> list[str | None]:
    match = REACT_ACTION.search(text)
    if match is None:
        return []
    name, raw = match.group(1), match.group(2).strip()
    try:
        args = json.loads(raw)
    except ValueError:
        args = raw
    if isinstance(args, dict):
        return [_python_call(name, args)]
    if not name.isidentifier():
        return [None]
    return [f"result = {name}({args!r})\nprint(result)"]


def extract_code(text: str) -> tuple[str | None, str | None]:
    """Find the code to run in an LLM response.

    Returns:
        (code, note): code is None when nothing usable was found; note is
        feedback for the LLM explaining how its answer was interpreted.
    """
    code, note = _from_python_blocks(text)
    if code is not None:
        return code, note
    for label, parser in (("XML", _from_xml), ("JSON", _from_json),
                          ("ReAct", _from_react)):
        calls = parser(text)
        if not calls:
            continue
        if None in calls:
            return None, (f"Your {label} tool call is malformed. "
                          + NO_CODE_NOTE)
        return "\n".join(c for c in calls if c), (
            f"Your {label} tool call was converted to Python. Prefer "
            "```python blocks.")
    return None, NO_CODE_NOTE
