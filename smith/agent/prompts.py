"""System and task prompts.

MBPP has a 6,000 input token budget shared by every turn, and each turn
resends the whole conversation: its prompt must stay tiny.
"""

from smith.models import MBPPTaskInput, SWEBenchTaskInput

FORMAT = """You solve tasks by writing Python code that runs in a sandbox.
Each turn: one short "Thought:" line, then exactly one code block:
```python
...
```<end_code>
You then receive "Observation:" with what your code printed. Never write
the Observation yourself. Call final_answer(...) as soon as you are done."""

MBPP_METHOD = """Test and submit in one turn when you can:
```python
import json
code = '''def add(a, b):
    return a + b'''
result = json.loads(run_tests(code, test_list))
print(result)
if result["success"]:
    final_answer(code)
```<end_code>"""

SWEBENCH_METHOD = """Fix the issue in the repository like a careful engineer:
1. Locate the relevant code (search_code, list_files, read_file).
2. Reproduce the bug before changing anything when possible.
3. Fix the root cause with minimal edits (edit_file).
4. Check with run_tests(); iterate until the failing tests pass.
5. final_answer(get_patch())
Print only what you need: long outputs waste your budget."""


def build_system_prompt(manual: str, benchmark: str) -> str:
    """Assemble the system prompt for 'mbpp' or 'swebench'."""
    method = MBPP_METHOD if benchmark == "mbpp" else SWEBENCH_METHOD
    return f"{FORMAT}\n\n{method}\n\n{manual}"


def mbpp_task_prompt(task: MBPPTaskInput) -> str:
    """Describe an MBPP task; test_list is predefined in the sandbox."""
    imports = "\n".join(task.test_imports)
    return (
        f"Task: {task.task_definition}\n"
        f"Signature: {task.function_definition}\n"
        + (f"Test imports:\n{imports}\n" if imports else "")
        + "Tests (available as the variable `test_list`):\n"
        + "\n".join(task.test_list)
    )


def mbpp_setup_code(task: MBPPTaskInput) -> str:
    """Code run once before the first turn to define `test_list`."""
    return f"test_list = {task.test_list!r}"


def swebench_task_prompt(task: SWEBenchTaskInput) -> str:
    """Describe a SWE-bench task."""
    hints = f"\n\nHints:\n{task.hints_text}" if task.hints_text else ""
    return (
        f"Repository: {task.repo} (checked out at /testbed)\n\n"
        f"Issue:\n{task.problem_statement}{hints}"
    )
