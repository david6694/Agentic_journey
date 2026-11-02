"""Code extraction from LLM responses."""

from smith.agent.extraction import NO_CODE_NOTE, extract_code


def test_python_block() -> None:
    code, note = extract_code("Thought: easy\n```python\nprint(1)\n```")
    assert code == "print(1)" and note is None


def test_unclosed_block_is_run_with_note() -> None:
    code, note = extract_code("Thought: x\n```python\nprint(1)\n")
    assert code == "print(1)"
    assert note is not None and "not closed" in note


def test_several_blocks_are_joined() -> None:
    code, _ = extract_code("```py\na = 1\n```\ntext\n```\nprint(a)\n```")
    assert code == "a = 1\n\nprint(a)"


def test_other_fenced_blocks_are_ignored() -> None:
    for tag in ("json", "text"):
        code, _ = extract_code(
            f"Thought: x\n```{tag}\n{{}}\n```\nnow:\n```python\nprint(1)\n```")
        assert code == "print(1)"


def test_xml_tool_call() -> None:
    text = ('<invoke name="read_file"><parameter name="filepath">'
            "/testbed/a'b.py</parameter></invoke>")
    code, note = extract_code(text)
    assert code == ("result = read_file(filepath=\"/testbed/a'b.py\")\n"
                    "print(result)")
    assert note is not None and "XML" in note


def test_json_tool_call() -> None:
    text = ('<tool_call>{"name": "search_code", "arguments": '
            '{"pattern": "def f", "file_pattern": "*.py"}}</tool_call>')
    code, _ = extract_code(text)
    assert code == ("result = search_code(pattern='def f', "
                    "file_pattern='*.py')\nprint(result)")


def test_react_call() -> None:
    code, _ = extract_code('Action: add\nAction Input: {"a": 1, "b": 2}')
    assert code == "result = add(a=1, b=2)\nprint(result)"


def test_values_cannot_inject_code() -> None:
    code, _ = extract_code(
        '<tool_call>{"name": "echo", "arguments": '
        '{"text": "\') ; import os #"}}</tool_call>')
    assert code == ("result = echo(text=\"') ; import os #\")\n"
                    "print(result)")


def test_malformed_call_is_reported() -> None:
    code, note = extract_code('<tool_call>{"name": "bad name"}</tool_call>')
    assert code is None
    assert note is not None and "malformed" in note


def test_nothing_found() -> None:
    assert extract_code("I think the answer is 42.") == (None, NO_CODE_NOTE)


def test_fence_inside_a_string_does_not_close_the_block() -> None:
    answer = (
        'Thought: Submit.\n```python\nfinal_answer("""```diff\n'
        '+    x = 1\n```""")\n```'
    )
    code, note = extract_code(answer)
    assert code == 'final_answer("""```diff\n+    x = 1\n```""")'
    assert note is None


def test_closing_fence_right_after_the_code() -> None:
    code, note = extract_code("```python\nprint(1)```<end_code>")
    assert code == "print(1)" and note is None
