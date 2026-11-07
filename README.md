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
# without --model-name: qwen/qwen3.8-27b (Groq), codestral-latest as fallback
uv run python -m agent_swebench --task-file task.json --output solution.json \
    --model-name "model/name" --provider-url "https://provider.api/v1"
uv run pytest tests
```

## System architecture

```
moulinette ──task.json──▶ agent_mbpp / agent_swebench (CLI)
                              │
                              ▼
           smith/agent/loop.py ◀──▶ smith/llm/client.py ──▶ LLM providers
                              │
                              ▼
           smith/sandbox/ (worker process) ◀──MCP──▶ mcp_tools_*.py
                                                      │ (SWE-bench)
                                                      ▼
                                              Docker container /testbed
```

- `smith/llm/`: OpenAI-compatible client. Several keys per provider
  (comma-separated), rotated on errors; retries follow `Retry-After` on HTTP
  429; an optional fallback client takes over when every retry fails.
  Providers and default model: `configs/providers.json`.
- `smith/agent/`: the loop, code extraction and prompts.
- `smith/sandbox/`: secure execution, MCP client, generated tool manual and
  the `uv run sandbox` REPL.
- `agent_mbpp/`, `agent_swebench/`: thin CLIs that build the sandbox, the
  MCP server and the LLM client, then write `solution.json`.
- `mcp_tools_mbpp.py`, `mcp_tools_swebench.py`: our MCP servers (stdio).

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

## Tool implementation details

Tools are served over MCP and discovered at runtime: their schemas become
Python stubs in the sandbox and are listed in the manual given to the LLM.

**MBPP** (`mcp_tools_mbpp.py`): `run_tests(code, test_list)` runs the code
and the assertions in a fresh `python -I` process (10s timeout), with
`import math` available like in the moulinette, and returns JSON
`{"success", "output"}`.

**SWE-bench** (`mcp_tools_swebench.py`): the 9 mandatory tools.
`read_file`, `edit_file` (rejects edits that break Python syntax),
`list_files`, `search_code`, `search_function_or_class_definition_in_code`
(AST-based) and `find_references` work on a local copy of `/testbed`;
`run_command`, `run_tests` and `get_patch`
(`git -c core.fileMode=false diff`) run inside the task's container.

- **Container lifecycle** (`agent_swebench/docker_env.py`): the image is
  pulled for `linux/amd64`, `/testbed` is copied to the host, and the
  container is removed on exit, on SIGINT/SIGTERM and, through a watchdog
  process, even after `kill -9`.
- **Synchronisation**: `edit_file` copies the file into the container (and
  undoes the local edit if that fails); after each container command, the
  files it changed are copied back. The container runs untrusted code, so
  only regular files within size limits are accepted, checked on the host.
- **What the LLM sees**: paths are shown as `/testbed/...`; `run_tests`
  returns only the part between the SWE-bench test markers; commands are
  stopped inside the container when they time out.
- **Isolation mode**: without a container, every tool works on
  `TESTBED_PATH`, as the moulinette's independent tool tests require.
- If the agent never calls `final_answer`, the repository patch is still
  submitted, with `success: false`.

## Benchmark results and analysis

Full report, with per-task tables, provider reliability, intermediary
metrics and ablations: [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).
Backing `solution.json` files are in `benchmarks/`.

- **SWE-bench** (3 tasks, `codestral-latest`): fixing the code-block
  extraction and submitting the repository patch as a fallback raised the
  result from 2/3 to 3/3.
- **MBPP, same 20 tasks for every model**: `qwen/qwen3.8-27b` (Groq) 19/20,
  `codestral-latest` (Mistral) 16/20, `openai/gpt-oss-120b` (Groq) 16/20.
  qwen's only failure was a Groq rate limit (HTTP 429); `codestral-latest`
  ran out of its input token budget on harder tasks.
- **Choices**: MBPP uses `qwen/qwen3.8-27b` with `codestral-latest` as
  fallback against Groq's quota; SWE-bench uses `codestral-latest`, whose
  requests fit Mistral's free tier better.
- **Free tiers** are the main operational risk: at about 2 requests a
  minute per key on Mistral, several keys per provider are needed.

## Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [PEP 578 – Runtime audit hooks](https://peps.python.org/pep-0578/)
- Python docs: [`multiprocessing`](https://docs.python.org/3.10/library/multiprocessing.html), [`ast`](https://docs.python.org/3.10/library/ast.html), [`resource`](https://docs.python.org/3.10/library/resource.html)
- [Executable Code Actions Elicit Better LLM Agents (CodeAct)](https://arxiv.org/abs/2402.01030)
- [SWE-bench](https://www.swebench.com/) and [MBPP](https://arxiv.org/abs/2108.07732)

**AI usage**: Claude was used as a coding assistant to implement the
sandbox, the agent loop and their tests, to review and harden the SWE-bench
tools, for security review, to script the benchmark runs, and to draft this
README.
Generated code was tested before being kept.
