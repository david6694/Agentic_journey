*This project has been created as part of the 42 curriculum by jerecaba, dgarrido.*

# Agent Smith

## Description

## Instructions

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

```bash
uv run sandbox [config.json] [--mcp-stdio "CMD" | --mcp-server URL] [--manual]
uv run pytest tests/sandbox
```

Known limit: `"{0.__class__}".format(x)` can read attributes as text (no
call possible).

## Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [PEP 578 – Runtime audit hooks](https://peps.python.org/pep-0578/)
- Python docs: [`multiprocessing`](https://docs.python.org/3.10/library/multiprocessing.html), [`ast`](https://docs.python.org/3.10/library/ast.html), [`resource`](https://docs.python.org/3.10/library/resource.html)
- [Executable Code Actions Elicit Better LLM Agents (CodeAct)](https://arxiv.org/abs/2402.01030)
- [SWE-bench](https://www.swebench.com/) and [MBPP](https://arxiv.org/abs/2108.07732)

**AI usage**: Claude was used as a coding assistant to implement the
sandbox and its tests, for security review, and to draft this README.
Generated code was tested before being kept.
