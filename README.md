*This project has been created as part of the 42 curriculum by jerecaba, dgarrido.*

# Agent Smith

## Description

Agent Smith is an autonomous coding agent built from scratch. An LLM
writes Python code, a secure sandbox runs it, and the result is fed back
until the agent calls `final_answer`. It is evaluated on MBPP (small
functions) and SWE-bench (fixing real GitHub issues), with tools served
over MCP.

## Instructions

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env                             # then add your API keys
uv run sandbox                                   # interactive sandbox
uv run sandbox sandbox_template.json             # custom configuration
uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
uv run sandbox --mcp-server <URL>                # MCP over HTTP
uv run python -m agent_mbpp --task-file task.json --output solution.json \
    --model-name "model/name" --provider-url "https://provider.api/v1"
uv run python -m agent_swebench --task-file task.json --output solution.json \
    --model-name "model/name" --provider-url "https://provider.api/v1"
uv run pytest tests
```

## Agent loop

`smith/agent/loop.py` repeats Thought → Code → Observation:

1. The LLM answers with a Thought and a code block; generation stops at
   `<end_code>` so it never invents the Observation.
2. `extraction.py` gets the code. XML, JSON and ReAct tool calls are
   converted to Python calls; the LLM is told when nothing usable was found.
3. The sandbox runs it; the shortened result goes back as `Observation:`.
4. It stops on `final_answer`, or before the iteration, token or time
   budget is exceeded. Every step is recorded in `SolutionOutput`.

## Sandbox design

The sandbox (`smith/sandbox/`) runs LLM-generated code in a separate worker
process, so timeouts and memory overuse are enforced by killing it.

```
Agent (parent)                              Worker (child)
 Sandbox.execute(code) ──── JSON pipe ────▶  exec(code, namespace)
   timeout, memory watchdog, MCP client      restricted builtins, audit hook
```

- **Persistent namespace**: variables survive between steps; the LLM is told
  when a restart wiped them.
- **JSON protocol**: worker messages are untrusted, never unpickled.
- **MCP in the parent**: tools become stubs in the worker and calls are
  relayed through the pipe, so the worker needs no network.
- **Security (stdlib only)**: AST check (no private/dunder access),
  restricted builtins, import allowlist, audit hook (files outside
  `allowed_directories`, sockets, subprocesses), `RLIMIT_AS` + RSS watchdog,
  timeout. Environment variables are cleared in the worker.
- **Explicit feedback**: truncated output, timeout, memory limit, restart and
  errors are reported to the LLM.

Known limit: `"{0.__class__}".format(x)` can read attributes as text (no
call possible).

## Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [PEP 578 – Runtime audit hooks](https://peps.python.org/pep-0578/)
- Python docs: [`multiprocessing`](https://docs.python.org/3.10/library/multiprocessing.html), [`ast`](https://docs.python.org/3.10/library/ast.html), [`resource`](https://docs.python.org/3.10/library/resource.html)
- [Executable Code Actions Elicit Better LLM Agents (CodeAct)](https://arxiv.org/abs/2402.01030)
- [SWE-bench](https://www.swebench.com/) and [MBPP](https://arxiv.org/abs/2108.07732)

**AI usage**: Claude was used as a coding assistant to implement the
sandbox, the agent loop and their tests, for security review, and to
draft this README.
Generated code was tested before being kept.
