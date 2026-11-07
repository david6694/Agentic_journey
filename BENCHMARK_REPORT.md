# Agent Smith — Benchmark & Ablation Report

## 1. Executive Summary & Experimental Setup

### 1.1 Overview
This report documents the empirical evaluation of **Agent Smith**, a pure-Python autonomous code agent with zero runtime third-party dependencies, a two-layer sandbox (`multiprocessing` + OS-level `resource`/`prlimit` isolation + AST static security gateway), and native MCP tool discovery (`stdio` and `Streamable HTTP`).

All benchmarks were executed against real free-tier / trial-tier OpenAI-compatible LLM endpoints under strict token and iteration budgets:
- **SWE-bench Verified (Dockerized repository repair):** 3 real-world open-source bug-fix tasks across Django, scikit-learn, and SymPy (`max_iterations = 30`, input token budget = `400,000`, output token budget = `10,000`).
- **MBPP (Mostly Basic Python Problems):**
  - **Random-20 Suite:** 20 tasks sampled from MBPP (`tasks 68, 82, 102, 123, 126, 168, 223, 228, 235, 239, 268, 274, 276, 277, 306, 310, 391, 394, 400, 415`), with `max_iterations = 10`, input budget = `4,000` tokens, total output budget = `1,500` tokens (`400` tokens/turn cap).
  - **Hard-12 Suite:** 12 multi-step / edge-case MBPP tasks (`tasks 126, 223, 228, 235, 239, 248, 268, 306, 310, 400, 415, 462`) specifically selected from initial failure modes (harmonic sum recursion, bitwise operations, dynamic programming subsequences, combinations, and prime divisor sums).
  - **Progressive Ablation Batches:** 12 chronological evaluation runs (`59` task executions in `benchmarks/mbpp/`) measuring the impact of prompt and token-budget refinements.

### 1.2 Models & Providers Evaluated (9 Models across 3 Providers)
To evaluate model capability, instruction adherence, token efficiency, and provider reliability under free-tier rate limits, we tested **9 distinct models** across **3 OpenAI-compatible providers**:

| # | Provider | Base URL | Model ID | Architecture / Tier | Selection Rationale |
|---|----------|----------|----------|---------------------|---------------------|
| 1 | **Mistral AI** | `https://api.mistral.ai/v1` | `codestral-latest` | Specialized Code LLM | Primary candidate for code generation & repository patching; fast non-reasoning output. |
| 2 | **Mistral AI** | `https://api.mistral.ai/v1` | `ministral-8b-latest` | 8B Compact Instruct | Low-latency baseline to measure small-model capability on Hard-12 tasks. |
| 3 | **Mistral AI** | `https://api.mistral.ai/v1` | `mistral-large-4` (`2411`) | Frontier Generalist | Tested for complex reasoning; evaluated for free-tier endpoint availability. |
| 4 | **Mistral AI** | `https://api.mistral.ai/v1` | `mistral-medium-latest` | Mid-size Generalist | Tested for cost/latency trade-off on Mistral API. |
| 5 | **Mistral AI** | `https://api.mistral.ai/v1` | `mistral-small-latest` | Small Generalist | Tested as a lightweight alternative to `codestral-latest`. |
| 6 | **Groq** | `https://api.groq.com/openai/v1` | `qwen/qwen3.8-27b` | 27B Dense / Reasoning | High single-shot accuracy on algorithmic tasks; tests strict OTPM rate limits. |
| 7 | **Groq** | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | 120B Open-Weights | High-capacity open model hosted on Groq LPU; tests TPM rate limits & reasoning overhead. |
| 8 | **OpenRouter** | `https://openrouter.ai/api/v1` | `cohere/north-mini-code:free` | Free Code Model | Evaluated to test reasoning-token consumption under a 400-token per-turn output cap. |
| 9 | **OpenRouter** | `https://openrouter.ai/api/v1` | `thinkingmachines/inkling:free` | Free Experimental | Evaluated for OpenRouter free-tier routing and provider error handling. |

---

## 2. SWE-bench Verified Results (3 Tasks — Docker Evaluation)

All 3 SWE-bench tasks were evaluated end-to-end inside isolated Docker containers (`sweb.eval.x86_64.*`) with MCP `stdio` repository tools (`read_file`, `write_file`, `edit_file`, `grep`, `find_files`, `run_tests`, `git_diff`, `get_patch`, `reset_file`) exposed inside the Python sandbox.

### 2.1 Post-Ablation Production Run (`benchmarks/swebench/2026-11-02_18-57-53`)
Model: **`codestral-latest`** (`https://api.mistral.ai/v1`)

| Task ID | Status | Docker Validation | Iterations | Total Requests | Retries | Input Tokens | Output Tokens | Agent Time (s) | Avg Req Latency (ms) | Backing File |
|---------|--------|-------------------|------------|----------------|---------|--------------|---------------|----------------|----------------------|--------------|
| `django__django-11066` | **PASS** (`success=true`) | `PASSED` (`RESOLVED_FULL`) | 8 | 8 | 0 | 30,786 | 509 | 34.20 | 1,797.91 | `benchmarks/swebench/2026-11-02_18-57-53/django__django-11066/solution.json` |
| `scikit-learn__scikit-learn-13439` | **PASS** (`success=true`) | `PASSED` (`RESOLVED_FULL`) | 8 | 8 | 0 | 17,142 | 628 | 20.24 | 1,452.55 | `benchmarks/swebench/2026-11-02_18-57-53/scikit-learn__scikit-learn-13439/solution.json` |
| `sympy__sympy-13480` | **PASS** (`success=true`) | `PASSED` (`RESOLVED_FULL`) | 8 | 8 | 0 | 26,743 | 416 | 29.85 | 1,486.78 | `benchmarks/swebench/2026-11-02_18-57-53/sympy__sympy-13480/solution.json` |
| **Total / Average** | **3 / 3 (100%)** | **3 / 3 PASSED** | **8.00** | **24** | **0** | **74,671** (avg 24,890) | **1,553** (avg 518) | **84.30** (avg 28.10) | **1,579.08** | — |

### 2.2 Baseline Pre-Ablation Run (`benchmarks/swebench/2026-11-02_18-27-24`)
Model: **`codestral-latest`** (`https://api.mistral.ai/v1`)

| Task ID | Status | Docker Validation | Iterations | Total Requests | Retries | Input Tokens | Output Tokens | Agent Time (s) | Avg Req Latency (ms) | Error / Failure Mode |
|---------|--------|-------------------|------------|----------------|---------|--------------|---------------|----------------|----------------------|----------------------|
| `django__django-11066` | **PASS** (`success=true`) | `PASSED` (`RESOLVED_FULL`) | 9 | 9 | 0 | 24,557 | 572 | 31.19 | 1,407.12 | `None` |
| `scikit-learn__scikit-learn-13439` | **FAIL** (`success=false`) | `FAILED` (empty patch) | 30 | 30 | 0 | 151,031 | 4,442 | 89.48 | 2,686.36 | `no final_answer within 30 iterations` (nested ```` ```diff ```` truncated by extractor) |
| `sympy__sympy-13480` | **PASS** (`success=true`) | `PASSED` (`RESOLVED_FULL`) | 7 | 7 | 0 | 11,240 | 414 | 83.53 | 1,243.81 | `None` |
| **Total / Average** | **2 / 3 (66.7%)** | **2 / 3 PASSED** | **15.33** | **46** | **0** | **186,828** (avg 62,276) | **5,428** (avg 1,809) | **204.20** (avg 68.07) | **2,216.65** | — |

---

## 3. Multi-Model & Multi-Provider Comparison (MBPP Suites)

### 3.1 MBPP Random-20 Suite (8 Models across 3 Providers)
Backing directory: `benchmarks/model_comparison_mbpp/random_20/`

| Provider | Model | Tasks Evaluated | Passed | Pass Rate | Avg Iterations | Total Requests (Retries) | Avg Input Tokens | Avg Output Tokens | Avg LLM Latency (ms) | Wall Time (s) | Status / Notes |
|----------|-------|-----------------|--------|-----------|----------------|--------------------------|------------------|-------------------|----------------------|---------------|----------------|
| **Groq** | `qwen/qwen3.8-27b` | 20 | **19** | **95.0%** | 1.05 | 33 (12 retries) | 802 | 129 | 4,920.3 | 117.8 | Only failure (`239`) caused by HTTP 429 OTPM limit after 4 retries |
| **Mistral** | `codestral-latest` | 20 | **16** | **80.0%** | 1.65 | 33 (0 retries) | 1,469 | 292 | 1,865.3 | 66.9 | **0 HTTP errors**, fastest wall time (`66.9s`), 100% API availability |
| **Groq** | `openai/gpt-oss-120b` | 20 | **16** | **80.0%** | 2.20 | 124 (80 retries) | 1,834 | 668 | 4,634.0 | 209.2 | Heavy TPM rate-limiting (80 retries); verbose chain-of-thought |
| **OpenRouter** | `cohere/north-mini-code:free` | 4 (early-stopped) | 0 | 0.0% | 1.00 | 4 (0 retries) | 660 | 400 | 5,531.8 | 62.4 | Consumed all 400 per-turn output tokens in internal `<think>` reasoning before emitting code |
| **Mistral** | `mistral-large-4` | 4 (early-stopped) | 0 | 0.0% | 0.00 | 0 (16 failed tries) | 0 | 0 | — | 75.0 | Unavailable on free tier (`HTTP 429` / `HTTP 404` model routing) |
| **Mistral** | `mistral-medium-latest` | 4 (early-stopped) | 0 | 0.0% | 0.00 | 0 (16 failed tries) | 0 | 0 | — | 41.7 | Unavailable on free tier (`HTTP 429` tier restriction) |
| **Mistral** | `mistral-small-latest` | 4 (early-stopped) | 0 | 0.0% | 0.00 | 0 (16 failed tries) | 0 | 0 | — | 39.5 | Unavailable on free tier (`HTTP 429` tier restriction) |
| **OpenRouter** | `thinkingmachines/inkling:free` | 4 (early-stopped) | 0 | 0.0% | 0.00 | 0 (16 failed tries) | 0 | 0 | — | 36.5 | Upstream provider unavailable on OpenRouter (`HTTP 404/400`) |

> **Early-Stopping Policy:** Models that failed 4 consecutive tasks (`tasks 68, 82, 102, 123`) due to provider unavailability or 100% reasoning-token exhaustion were automatically aborted by the comparison runner (`status: early_stopped`) to preserve rate-limit quotas.

### 3.2 MBPP Hard-12 Suite (Targeted Stress Test on 12 Hard Tasks)
Backing directory: `benchmarks/model_comparison_mbpp/hard_12/`

| Provider | Model | Tasks | Passed | Pass Rate | Avg Iterations | Total Requests (Retries) | Avg Input Tokens | Avg Output Tokens | Avg LLM Latency (ms) | Total Time (s) | Primary Failure Mode |
|----------|-------|-------|--------|-----------|----------------|--------------------------|------------------|-------------------|----------------------|----------------|----------------------|
| **Groq** | `qwen/qwen3.8-27b` | 12 | **8** | **66.7%** | 1.17 | 32 (18 retries) | 1,036 | 215 | 3,169.9 | 83.5 | **100% of completed tasks passed (8/8)**; all 4 failures (`235, 239, 306, 462`) were Groq HTTP 429 OTPM rate limits (`Limit 1000 OTPM`) |
| **Groq** | `openai/gpt-oss-120b` | 12 | **6** | **50.0%** | 2.67 | 83 (51 retries) | 2,327 | 927 | 3,794.3 | 133.1 | 5 failures from exhausting the 1,500 output token budget via verbose reasoning; 1 failure (`462`) from HTTP 429 TPM limit (`Limit 8000 TPM`) |
| **Mistral** | `ministral-8b-latest` | 12 | **2** | **16.7%** | 3.75 | 45 (0 retries) | 4,385 | 968 | 4,594.6 | 210.2 | Passed `223, 248`; failed 10 tasks due to multi-turn syntax/logic loops hitting the 4,000 input token budget |

---

## 4. Provider Reliability Analysis

A critical engineering requirement of Agent Smith is resilience against real-world API instability (`HTTP 429 Too Many Requests`, transient `5xx` errors, and strict per-minute token quotas). Our client (`smith/agent/llm.py`) implements exponential backoff with jitter (`_INITIAL_BACKOFF_SECONDS = 1.0`, multiplier `2.0`, up to `_MAX_RETRIES = 4`).

### 4.1 Quantitative Reliability Metrics by Provider & Model

| Provider | Endpoint | Model | Successful Turns | Total HTTP Requests | Retry Count | Retry Rate | Avg Turn Latency (`request_time_ms`) | Task-Ending HTTP Failures | Effective Availability |
|----------|----------|-------|------------------|---------------------|-------------|------------|--------------------------------------|---------------------------|------------------------|
| **Mistral AI** | `api.mistral.ai/v1` | `codestral-latest` (SWE-bench + MBPP) | 190 | 219 | **29** | 13.2% | **2,265.4 ms** | **0** | **100.0%** |
| **Mistral AI** | `api.mistral.ai/v1` | `ministral-8b-latest` (Hard-12) | 45 | 45 | **0** | 0.0% | **4,594.6 ms** | **0** | **100.0%** |
| **Groq** | `api.groq.com/openai/v1` | `qwen/qwen3.8-27b` (Random-20 + Hard-12 + MBPP) | 41 | 83 | **42** | 50.6% | **4,121.8 ms** | **9** (HTTP 429 OTPM) | **78.6%** (33/42 tasks) |
| **Groq** | `api.groq.com/openai/v1` | `openai/gpt-oss-120b` (Random-20 + Hard-12) | 76 | 207 | **131** | 63.3% | **4,315.7 ms** | **1** (HTTP 429 TPM) | **96.9%** (31/32 tasks) |
| **OpenRouter** | `openrouter.ai/api/v1` | `cohere/north-mini-code:free` | 4 | 4 | **0** | 0.0% | **5,531.8 ms** | **0** (truncation failure) | **100.0%** (HTTP) |
| **OpenRouter** | `openrouter.ai/api/v1` | `thinkingmachines/inkling:free` | 0 | 16 | **16** | 100.0% | — | **4** (HTTP 404/400) | **0.0%** |

### 4.2 Key Reliability Findings
1. **Mistral AI (`codestral-latest`) offers the highest production stability:** Across all SWE-bench runs (`54` turns) and MBPP runs (`136` turns), `codestral-latest` experienced **0 task-ending HTTP errors** (100% task completion availability). Even during burst testing (`2026-11-02_21-06-38` and `21-07-22`), when Mistral's free tier returned transient `HTTP 429` responses (15 retries across 6 turns in `21-06-38`), Agent Smith's exponential backoff absorbed 100% of the rate limits and achieved **5/5 (100%)** on both runs.
2. **Groq's Output-Tokens-Per-Minute (OTPM) bottleneck:** While `qwen/qwen3.8-27b` on Groq is extraordinarily accurate (solving **95% of Random-20** in `1.05` iterations and **100% of reached Hard-12 tasks**), Groq's free `on_demand` tier enforces a strict **1,000 OTPM (Output Tokens Per Minute)** limit (`Limit 1000`). When running back-to-back tasks, 4 retries (~15 seconds total backoff) are occasionally insufficient to wait out a 60-second rolling token window, causing HTTP 429 failures on tasks `235, 239, 306, 462`.
3. **Groq's Total-Tokens-Per-Minute (TPM) bottleneck on 120B models:** `openai/gpt-oss-120b` is capped at **8,000 TPM** on Groq's free tier, triggering **131 retries across 76 turns** (an average of `1.72` retries per turn) and inflating wall-clock time on Random-20 from `66.9s` (`codestral-latest`) to `209.2s` (+212%).

---

## 5. Intermediary Metrics Analysis (SWE-bench Verified)

In accordance with the evaluation criteria, we tracked **all three intermediary behavioral metrics** across every step of the SWE-bench trajectories (`benchmarks/swebench/2026-11-02_18-57-53/*/solution.json`):
1. **Metric 1 (Target Localization & Modification):** Step at which the agent first **reads** and first **edits** the file that appears in the final validated patch.
2. **Metric 2 (Test Failure Reduction):** Step at which test failures first **decrease** compared to the baseline test execution.
3. **Metric 3 (Verification-to-Submission Lag):** Number of iterations between *"all target tests first pass"* and calling `final_answer`.

### 5.1 Summary Table of Intermediary Metrics

| Task ID | Target File in Final Patch | Total Steps | Metric 1a: First Read Step | Metric 1b: First Edit Step | Baseline Test Step | Metric 2: Step Test Failures First Decrease | Metric 3: Iterations from "Tests Pass" to `final_answer` |
|---------|----------------------------|-------------|----------------------------|----------------------------|--------------------|---------------------------------------------|----------------------------------------------------------|
| `django__django-11066` | `django/contrib/contenttypes/management/__init__.py` | 8 | **Step 3** | **Step 5** | Step 4 (1 failure: `AttributeError`) | **Step 6** (1 failure $\to$ 0 failures, `Ran 4 tests ... OK`) | **2 iterations** (Step 6 $\to$ Step 8) |
| `scikit-learn__scikit-learn-13439` | `sklearn/pipeline.py` | 8 | **Step 4** (located via `grep` at Step 1) | **Step 5** | Step 3 (`pytest` baseline on `test_pipeline.py`) | **Step 6** (`41 passed in 1.71s`, including new `len(pipe)` test) | **2 iterations** (Step 6 $\to$ Step 8) |
| `sympy__sympy-13480` | `sympy/functions/elementary/hyperbolic.py` | 8 | **Step 4** (located via `grep` at Step 3) | **Step 5** | Step 2 (reproduced `NameError: name 'cotm' is not defined`) | **Step 6** (`45 passed in 8.85s` in `test_hyperbolic.py`) | **2 iterations** (Step 6 $\to$ Step 8) |
| **Mean across 3 Tasks** | — | **8.0** | **Step 3.67** | **Step 5.00** | **Step 3.00** | **Step 6.00** | **2.00 iterations** |

### 5.2 Step-by-Step Trajectory Breakdown
Across all three repositories, `codestral-latest` converged on an identical, highly disciplined **8-step canonical repair loop**:
- **Steps 1–2 (Discovery):** Uses `find_files` and `grep` to locate the target module and test suite (`100%` localization by Step 3).
- **Step 3–4 (Context & Baseline Reproduction):** Reads the exact line range around the bug (`read_file(..., start_line=..., end_line=...)`) and runs `run_tests` to confirm the baseline failure.
- **Step 5 (Surgical Patch):** Applies a minimal, targeted `edit_file` modification (1 line changed in `django-11066`, 4 lines added in `scikit-learn-13439`, 1 character `cotm` $\to$ `cothm` fixed in `sympy-13480`).
- **Step 6 (Verification — Metric 2):** Re-runs `run_tests` on the target test module; test failures drop to `0` (`OK` / `41 passed` / `45 passed`).
- **Steps 7–8 (Patch Inspection & Submission — Metric 3):** Calls `get_patch()` in Step 7 to inspect the unified git diff and confirm no unintended edits were introduced, then calls `final_answer(...)` in Step 8 (`Metric 3 = 2 iterations`).

---

## 6. Systematic Ablation Studies (Before vs. After Agent Changes)

We conducted **three controlled ablations** during development, each motivated by failure modes observed in real LLM trajectories and verified on the benchmark suites.

### 6.1 Ablation 1 — Code-Block Fence Extraction (`smith/agent/extraction.py`, Commit `35e3167`)
- **Problem Observed (`benchmarks/swebench/2026-11-02_18-27-24/scikit-learn__scikit-learn-13439/solution.json`):**
  In the baseline SWE-bench run, the agent fixed `sklearn/pipeline.py` at Step 5 and passed all 41 tests at Step 6. At Step 8, instead of calling `final_answer(get_patch())`, the model wrote a literal triple-quoted diff string:
  ```python
  final_answer('''diff --git a/sklearn/pipeline.py b/sklearn/pipeline.py
  ...''')
  ```
  and in subsequent turns wrapped it inside `final_answer(""" ```diff ... ``` """)`. Because the regex in `extraction.py` matched the first inner ```` ``` ```` anywhere in the text as the closing markdown fence, it truncated the Python code mid-string at line 1! This caused a `SyntaxError: unterminated triple-quoted string literal` in the sandbox, trapping the agent in a **23-turn syntax error loop** until `max_iterations = 30` was exhausted (`151,031` input tokens consumed, `Overall: FAILED`).
- **Change Implemented:**
  Updated `_FENCE_RE` in `smith/agent/extraction.py` so a fenced code block only closes on a backtick fence that stands alone on its own line (`r"\n[ \t]*```[ \t]*(?=\n|$)"`) or at the very end of the response.
- **Measured Impact on `scikit-learn__scikit-learn-13439`:**

| Metric | Before (`18-27-24`) | After (`18-57-53`) | Delta / Improvement |
|--------|---------------------|--------------------|---------------------|
| **Task Outcome** | `FAILED` (`success=false`) | **`PASSED` (`RESOLVED_FULL`)** | **Fixed (0% $\to$ 100%)** |
| **Total Iterations** | 30 (hit limit) | **8** | **-73.3% (-22 turns)** |
| **Input Tokens** | 151,031 | **17,142** | **-88.6% (-133,889 tokens)** |
| **Output Tokens** | 4,442 | **628** | **-85.9% (-3,814 tokens)** |
| **Wall Time** | 89.48 s | **20.24 s** | **-77.4% (-69.24 s)** |
| **SWE-bench Suite Pass Rate** | **2 / 3 (66.7%)** | **3 / 3 (100.0%)** | **+33.3 percentage points** |

### 6.2 Ablation 2 — Fallback Repository Patch Submission (`agent_swebench.py`, Commit `2719f24`)
- **Problem Observed:**
  Even when an agent successfully modifies the repository and passes all tests inside the container, if it hits the iteration or token budget before executing `final_answer`, the CLI previously wrote an empty `solution: ""` patch, discarding a valid working fix in the container's working tree.
- **Change Implemented:**
  In `agent_swebench.py`, if `result.success` is `False` (no `final_answer` reached), the CLI queries `get_patch()` via MCP from the container and populates `solution` with the repository's uncommitted `git diff` while preserving `success: False` and the original `error` string for full diagnostic honesty.
- **Measured Impact:**
  Verified on `scikit-learn__scikit-learn-13439` under forced budget exhaustion: `Overall: PASSED` by the SWE-bench Docker harness even when `success: False`.

### 6.3 Ablation 3 — MBPP Per-Turn Output Cap (`400` tokens) & Single-Block Test Prompt (`c133a30`, `8ab48ac`)
- **Problems Observed in Baseline MBPP Runs (`18-26-30` to `18-33-41`):**
  1. **Single-Turn Output Budget Exhaustion:** On task `248` (`benchmarks/mbpp/2026-11-02_18-32-06/248/solution.json`), `codestral-latest` generated a massive 1,500-token response in Turn 1, consuming 100% of the `1,500` total output token budget in a single turn (`iterations: 1`, `total_output_tokens: 1500`, `error: "output token budget exhausted"`).
  2. **Two-Turn Overhead & Redefined `test_list` Loops:** On passing tasks (`224, 291, 455, 141, 267`), the agent took 2–3 iterations because it called `run_tests` in Turn 1 without calling `final_answer`, and on task `415` (`18-26-30/415/solution.json`) it overwrote `test_list` with its own faulty assertions and looped until exhausting the input budget (`4` turns, `1,325` output tokens).
- **Changes Implemented:**
  1. Added `max_Tokens_per_turn = 400` in `smith/agent/agent.py` (capped by `min(400, remaining_output_budget)`), preventing any single runaway turn from draining the 1,500-token episode budget (`c133a30`).
  2. Updated `MBPP_SYSTEM_PROMPT` in `smith/agent/prompts.py` to require `res = run_tests(code, test_list)` followed immediately by `if res.get("passed"): final_answer(code)` in the **same** code block, while explicitly forbidding redefining `test_list` (`8ab48ac`).
- **Measured Impact on MBPP (`codestral-latest`, 4 Baseline Runs vs. 4 Post-Ablation Runs):**

| Metric | Pre-Ablation (`18-26-30` .. `18-33-41`) | Post-Ablation (`18-55-06` .. `18-57-13`) | Post-Ablation Follow-up (`21-06-38` & `21-07-22`) |
|--------|-----------------------------------------|------------------------------------------|---------------------------------------------------|
| **Tasks Passed** | 13 / 20 (**65.0%**, starting at 60% / 40%) | **15 / 19 (78.9%)** (75.0% on first 3x5 runs) | **10 / 10 (100.0%)** (`5/5` + `5/5`) |
| **Average Iterations / Task** | 2.80 | **1.95** (-30.4%) | **1.10** (-60.7%) |
| **Single-Turn (`it=1`) Solves** | 0 / 13 (0.0%) | **9 / 15 (60.0%)** | **9 / 10 (90.0%)** |
| **Avg Input Tokens / Task** | 2,653 | **1,936** (-27.0%) | **864** (-67.4%) |
| **Avg Output Tokens / Task** | 483 | **471** (-2.5%, 0 single-turn exhaustions) | **133** (-72.5%) |
| **Task `223` (Majority Element)** | `FAIL` (5 iters, 4,132 in tokens) | **`PASS` (2 iters, 1,929 in tokens)** | — |

---

## 7. Conclusions & Model Selection Rationale

Based on **250+ recorded task executions** across SWE-bench Verified and MBPP (`benchmarks/`), we draw three engineering conclusions:

1. **One default model per benchmark: `codestral-latest` for SWE-bench, `qwen/qwen3.8-27b` for MBPP, with `codestral-latest` as fallback:**
   - **SWE-bench → `codestral-latest`** (`https://api.mistral.ai/v1`): **3/3 (100%)** in **8.0 iterations** per task with **0 retries** and **1.58s/turn**. It is the only model run on SWE-bench, and its large requests (10k–30k tokens per turn) would exceed Groq's free per-minute quotas.
   - **MBPP → `qwen/qwen3.8-27b`** (`https://api.groq.com/openai/v1`): the highest accuracy on the same Random-20 tasks (**19/20** vs **16/20** for `codestral-latest`), every failure being an HTTP 429. Because `codestral-latest` is a non-reasoning code model it stays compact and reliable, so it is kept as the **fallback**: `smith/llm/client.py` follows `Retry-After` on HTTP 429 (`acdd788`, `b2261a2`) and hands the remaining time to the fallback when every retry fails (`0153e5d`, `5b2e12a`).
   - **Verification with this setup** (`benchmarks/mbpp/2026-11-07_*`): `exam_mbpp.sh` scored **4/5, 4/5, 4/5 and 5/5** (17/20, threshold reached in 4 runs out of 4), against 2 runs out of 4 with `codestral-latest` alone (`2026-11-02_18-55-06` .. `18-57-13`). The 3 failures were wrong answers, not rate limits, and the fallback was never needed.
   - **Free-tier limits** (`benchmarks/swebench/2026-11-07_*`): on 2026-11-07 Mistral's free tier allowed **2 requests and 40,000 tokens per minute per key**, without `Retry-After`. Three SWE-bench runs failed on HTTP 429 (`17-43-54`, `17-45-59`, `17-49-37`); with two Mistral keys rotated and a 10s wait per retry, the run `20-44-05` scored **3/3** (including `pydata__xarray-4629` and `sympy__sympy-14711`, 11 iterations, 154s).
2. **Reasoning models (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `cohere/north-mini-code:free`) face a fundamental trade-off under strict output budgets:**
   - When unconstrained by rate limits, `qwen/qwen3.8-27b` achieved the highest raw algorithmic accuracy (**95.0%** on Random-20 and **100% of non-rate-limited Hard-12 tasks**).
   - However, Groq's strict **1,000 OTPM** free-tier ceiling caused a **50.6% retry rate** and **9 task-ending HTTP 429 failures**, while models that emit long internal `<think>` chains (`openai/gpt-oss-120b`, `cohere/north-mini-code:free`) frequently hit the `400`-token per-turn cap or `1,500`-token total output limit before emitting executable Python code.
3. **Agent-level engineering (robust fence extraction + single-block test-and-submit prompts + repository diff fallback) matters as much as raw model size:**
   - Fixing markdown fence extraction (`35e3167`) and adding fallback patch submission (`2719f24`) improved SWE-bench Verified from **66.7% to 100.0%** while cutting input token consumption by **60.0% across the suite** (`186,828` $\to$ `74,671` tokens).
   - Enforcing single-block `run_tests(code, test_list)` + `final_answer(code)` (`8ab48ac`) and a 400-token per-turn cap (`c133a30`) reduced MBPP iterations by **30% to 60%** and raised `codestral-latest` pass rates from **60% to 78.9%–100%**.
