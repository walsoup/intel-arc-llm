# AGENTS.md — Repository Guide & Operating Manual

> **Note for Future AI Agents & Maintainers**: Read this file first before making changes, running background processes, or adding models.

---

## 1. Project Overview

This repository is a self-contained, portable local LLM inference environment powered by `llama.cpp` (`llama-server`) running on Windows x64, optimized for **Intel Arc iGPU (Vulkan)** with dynamic model switching.

* **Primary Web UI / API**: `http://127.0.0.1:8080` (OpenAI-compatible endpoints at `/v1/*`)
* **Core Server Mode**: **Router Mode** (`--models-dir .\Models --models-max 1`)
* **Primary Engine**: `bin-prism-vulkan` (Build 10683, Vulkan backend, Classic Web UI)
* **Secondary / Backup Engine**: `bin-prism-cpu` (Build 10683, CPU backend)

---

## 2. Directory Layout

```
noble-hawking/
 ├── Models/                  # All GGUF model files (auto-discovered by router)
 │    ├── Bonsai-2-27B.gguf
 │    ├── Qwable-9B-Claude-Fable-5-OBLITERATED.Q4_K_M.gguf
 │    ├── Qwen3.5-0.8B-Dark-Genius.Q4_K_M.gguf
 │    ├── Qwen3.5-9B-Fable-5-Self-Distill.q3_k_m.gguf
 │    ├── Qwen3.5-9B-The-Defiant-Fable-DARK-ROAST-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP.Q3_K_M.gguf
 │    ├── Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-Q8_0.gguf
 │    ├── Qwen3.8-2B-Q8_0.gguf
 │    ├── Qwythos-9B-v2-Heretic-Abliterated-Q2_K.gguf
 │    ├── Qwythos-9B-v2-MTP-Q4_K_M.gguf
 │    ├── qwythos_heretic_header.bak      # Backup of original Heretic Q2_K header
 │    └── qwythos_template_original.bak   # Backup of original unpatched template
 ├── bin-linux-vulkan/        # Native Linux llama-server engine (Intel Arc Vulkan, Build 11385)
 ├── bin-linux-cpu/           # Native Linux CPU fallback engine (Build 11385)
 ├── bin-prism-vulkan/        # Windows active llama-server engine (Intel Arc Vulkan + Classic UI)
 ├── bin-prism-cpu/           # Windows CPU fallback engine
 ├── scripts/                 # Standardized datasets and diagnostic utilities
 │    ├── tune-hardware.sh                # Root script: locks GPU 2200MHz, CPU perf, THP
 │    ├── benchmark_dataset_30.json       # 30-task standardized tri-domain benchmark suite
 │    ├── benchmark_strongreject.py       # Safety / refusal evaluation script
 │    └── automated_runner/               # Automated runner fallback utilities
 │         └── benchmark_runner.py
 ├── webui/                   # Standalone lightweight custom HTML client (optional)
 ├── chat_template.jinja      # Reference clean Jinja template with thinking support
 ├── start.bat                # 1-Click Windows launcher: starts server & opens browser
 ├── start-server.bat         # Windows router server launcher
 ├── start-server.sh          # Native Linux router launcher with MTP/N-gram speculative acceleration
 ├── restart-server.bat       # Windows quick reload
 ├── start-webui.bat          # Browser shortcut to http://127.0.0.1:8080
 ├── AGENTS.md                # This manual
 ├── README.md                # Upstream llama.cpp CLI documentation
 └── README-dev.md            # llama-server internal architecture docs
```

---

## 3. Key Architectural Decisions & Context

### Engine Selection: Why `bin-prism-vulkan`?
* **Classic Web UI Layout**: In `bin-prism-vulkan` (Build 10683), the Reasoning Effort selector (`Default`, `Off`, `Low`, `Medium`, `High`, `Max`) is cleanly integrated inside the **`+` (attach/actions) popover menu** right beside the chat input prompt. Upstream newer builds (such as `bin-vulkan` Build 10964) removed this from the `+` menu and relocated it into the top header. The user specifically prefers the classic layout.
* **Intel Arc Vulkan Acceleration**: Uses `ggml-vulkan.dll` with `-fa on` (Flash Attention) for fast generation and low memory overhead on Intel Arc integrated/discrete GPUs.

### Multi-Model Router Mode
* Launched via `llama-server.exe --models-dir .\Models --models-max 1 ...`
* **Dynamic Model Switching**: Any `.gguf` file placed in `Models/` is automatically discovered and listed in the Web UI's model picker.
* **VRAM Safety (`--models-max 1`)**: Critical for iGPU setups. When a user selects another model in the Web UI, the previous model is automatically unloaded before the new one loads, preventing system RAM / VRAM exhaustion.
* **Inherited Defaults**: The router passes `-ngl 99 -fa on -c 16384 -t 6` to every dynamically spawned model instance.

---

## 4. Model-Specific Quirks & Fixes

### Qwythos-9B (`Qwythos-9B-v2-MTP-Q4_K_M.gguf`)
* **The Problem**: The original GGUF metadata contained a hardcoded identity line in `tokenizer.chat_template`:
  `{%- set qwythos_identity = "You are Qwythos, a model created by Empero AI. Only bring up your identity if the user asks." %}`
  This was injected into `<|im_start|>system` on *every single request*. Because Qwythos has base Qwen weights, this forced injection triggered constant reasoning loops where the model debated whether it was Qwen or Qwythos, wasting output tokens on identity crises.
* **The Fix**: The template inside the GGUF header was patched byte-for-byte in-place (offset `10934640`, 7,950 bytes) using Qwable's clean system prompt logic (no forced identity injection, clean thinking tags, and reasoning effort toggle support).
* **Backup**: The original 7,950-byte template is preserved at `Models/qwythos_template_original.bak`.
* **Guideline for Agents**: Do **not** inject hardcoded identities or fake organization prompts into system prompts or model templates.

### Qwythos-9B Heretic Abliterated (`Qwythos-9B-v2-Heretic-Abliterated-Q2_K.gguf`)
* **The Problem & Patch**: The abliterated Q2_K conversion had 427 tensors (layers 0 through 31, with the 15 MTP layer-32 tensors stripped), but its metadata retained `qwen35.block_count = 33`, `qwen35.nextn_predict_layers = 1`, and `qwen35.attention.recurrent_layers` of length 33. This was patched in the header:
  1. `qwen35.block_count`: 33 -> 32 (offset `687`, 4-byte uint32).
  2. `qwen35.nextn_predict_layers`: 1 -> 0 (offset `1336`, 4-byte uint32).
  3. Renamed `qwen35.attention.recurrent_layers` to `qwen35.attention.recurrent_ignore` (offset `1539`, 33 bytes) so llama.cpp gracefully falls back to `qwen35.full_attention_interval = 4` across all 32 layers.
  4. Patched `tokenizer.chat_template` in-place byte-for-byte (offset `10935076`, 7,950 bytes) with the clean reasoning template from Qwable.
* **Quantization Collapse Diagnosis**: Despite the header fix allowing the model to load, the model scored **0.14 / 10.00** across benchmarks. Abliterating attention directions strips weight redundancy; compressing this perturbed 9B recurrent/transformer hybrid down to ~2.5 bits (`Q2_K`) causes severe mathematical entropy loss. The output logits collapse into `<|im_end|>` immediately, or loop endlessly if EOS is penalized. This is an intrinsic quantization loss that **cannot be resolved via metadata or prompt engineering**.

### Qwable-9B (`Qwable-9B-Claude-Fable-5-OBLITERATED.Q4_K_M.gguf`)
* Gold-standard template implementation. Correctly handles thinking tags (`<think>...</think>`), multi-turn query indexing, and structured function/tool calling. Yields high inference speed (~6.6 t/s).

### Qwen3.5-9B The Defiant Fable NEO-IMATRIX (`Qwen3.5-9B-The-Defiant-Fable-DARK-ROAST...Q3_K_M.gguf`)
* Outstanding performance (overall score **8.68 / 10.00**). The NEO Importance Matrix (imatrix) preserves attention projection matrices during low-bit quantization, allowing the 3-bit quant to outperform standard 4-bit models in complex coding, tool calls, and creative prose.

---

## 5. Model Benchmarking Protocol (AI Agent as Grader)

> **CRITICAL RULE**: The user explicitly requires the **AI agent to be the evaluator and grader** for all model benchmarks, scoring completions with fine precision down to **0.01** (scale 0.00 to 10.00). Do **not** delegate grading to automated regex or heuristic scripts.

### 5.1 Standard Benchmark Dataset
The standardized benchmark suite is located at:
* [`scripts/benchmark_dataset_30.json`](scripts/benchmark_dataset_30.json)

It contains **30 curated tasks** divided equally across three core domains:
1. **Coding (10 tasks, IDs `code-01` to `code-10`)**:
   * Algorithmic design (LRU Cache, sliding window, cyclic graph detection).
   * Concurrency, asyncio, rate limiters, token bucket.
   * TypeScript generics, complex SQL window queries, debug refactors.
2. **Tool Calling (10 tasks, IDs `tool-01` to `tool-10`)**:
   * Multi-argument function calling, strict JSON schema validation.
   * Handling missing required parameters (clarifying with user vs. hallucinating fake inputs).
   * Negative testing (answering factual questions directly without invoking irrelevant tools).
3. **Creative Writing (10 tasks, IDs `creative-01` to `creative-10`)**:
   * Hard negative constraints (strict word count limits, lipograms banning common letters like 'e', non-cliché sci-fi metaphors, philosophical dialogues).

### 5.2 Benchmark Execution Procedure for AI Agents
When the user asks to benchmark or compare models:
1. **Load Tasks**: Read `scripts/benchmark_dataset_30.json`.
2. **Run Inference**:
   * Send HTTP POST requests to `http://127.0.0.1:8080/v1/chat/completions`.
   * Pass `"model": "<model_name_or_file>"` (the router will auto-swap the model).
   * Recommended parameters: `temperature: 0.2`, `max_tokens: 2048` (or task-specific limit).
   * Record completion text, token counts (`usage.completion_tokens`), and response latency.
3. **Perform AI Agent Grading**:
   * Inspect each output against the task's ground truth, test assertions, or negative constraints.
   * Assign a score between **0.00 and 10.00** with precision down to **0.01** using the rubrics:
     * **Coding Rubric**: Correctness & logic (4.0), edge case & bounds handling (2.5), language idioms/clean code (2.0), output efficiency/lack of reasoning loops (1.5).
     * **Tool Calling Rubric**: Valid JSON schema & parameter adherence (5.0), negative constraint / missing argument discipline (3.0), hallucination resistance (2.0).
     * **Creative Writing Rubric**: Hard negative constraint compliance (4.0), original literary quality & tone (3.0), thematic coherence & formatting (3.0).
4. **Compile & Present Results**:
   * Present domain averages and overall weighted scores.
   * Highlight strengths, failure modes, tool-calling edge cases, and token throughput (tokens/sec).

### 5.3 Historical Benchmark Reference Table (30 Tasks Graded by AI)

| Model | Quant | Coding (/10) | Tools (/10) | Creative (/10) | **Overall** | Speed | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Defiant Fable 9B DARK-ROAST** | Q3_K_M | **7.95** | **9.73** | **8.36** | **8.68** | 5.86 t/s | NEO-IMATRIX preserves attention; minimal thinking overhead; superior prose. |
| **Qwythos-9B-v2 MTP** | Q4_K_M | **7.79** | **9.71** | **8.15** | **8.55** | 4.89 t/s | Excellent tool calling and logic; longer reasoning traces. |
| **Qwable-9B Claude-Fable-5** | Q4_K_M | **7.99** | **9.16** | **7.83** | **8.33** | **6.59 t/s** | Fastest model; strong coding; slight hallucination on missing tool parameters. |
| **Qwythos-9B-v2 Heretic Abliterated** | Q2_K | **0.11** | **0.11** | **0.20** | **0.14** | 2.12 t/s | Degenerate entropy collapse from 2-bit quantization on abliterated weights. |

---

## 7. Native Linux Execution & Hardware Throughput Optimization

### 7.1 Native Linux Engines
* **Primary Linux Engine**: `bin-linux-vulkan/` (Build 11385, Native x86_64, Mesa ANV Intel Arc Vulkan). Delivers +26% prefill throughput (56.7 t/s) and +16% decode throughput (6.84 t/s) over Wine emulation.
* **CPU Linux Engine**: `bin-linux-cpu/` (Build 11385, Alder Lake AVX2/VNNI, 7.05 t/s).
* **1-Click Linux Launcher**: [`start-server.sh`](start-server.sh) (runs on `http://127.0.0.1:8080` with `--spec-type draft-mtp,ngram-mod`).

### 7.2 Hardware Performance Tuning (`scripts/tune-hardware.sh`)
Execute `sudo ./scripts/tune-hardware.sh` to lock system hardware into peak performance:
* **CPU Governor**: Pinned to `performance` on all cores (4.2–4.4 GHz, avoiding C-state latency).
* **Intel Arc GPU (`gt0`)**: Frequency locked to max boost `2200 MHz` (`rps_min_freq_mhz = 2200`), eliminating layer-dispatch downclocking.
* **Transparent Huge Pages**: Set to `always` to maximize TLB hit rate across 5.3 GB sequential sweeps.
* **Virtual Memory**: `vm.swappiness = 10`.

### 7.3 Memory Wall & Speculative Acceleration Strategy
Autoregressive decoding of 9B models on 128-bit DDR5-5600 is fundamentally capped by memory bandwidth (~38.4 GB/s real-world limit, yielding ~6.8 TPS). To reach **15–22+ TPS**:
1. **Native MTP Self-Speculation**: Models retaining Layer 32 MTP weights (`nextn_predict_layers = 1`) self-speculate without external draft models at zero additional DRAM bandwidth cost.
2. **N-Gram Self-Speculation**: In-memory context hash matching (`--spec-type ngram-mod`) yields **14–18.5 TPS** on code and structured output with 0 VRAM penalty.
3. **Never Use Standalone Draft Models (`draft-simple`) on UMA iGPUs**: Streaming two models across a shared 128-bit bus causes memory bus thrashing and drops throughput to < 3 TPS.

---

## 8. Rules for Future AI Agents

1. **Keep the Root Directory Clean**:
   * Do **NOT** leave scratch python scripts, decompress dumps, test logs, or temporary batch files in the root.
   * Do **NOT** leave `.zip` archives loose in the project folder.
   * All diagnostic scripts must go in `scripts/` (or subfolders) or temporary directories, and temporary files must be cleaned up before finishing.
2. **Model Placement**:
   * All new GGUF models must go into `Models/`.
   * Never place multi-gigabyte models in the root folder.
3. **Preserve `bin-prism-vulkan` and `bin-prism-cpu`**:
   * Do not replace or overwrite `bin-prism-vulkan` without explicit user permission, as the user depends on its specific Web UI build and behavior.
4. **Testing Ports**:
   * The user's primary server runs on port **8080**.
   * When testing server commands in background tasks or scripts, use a non-conflicting port (e.g., `8089`) and **always kill the test process** when done so memory and ports remain free.
5. **Launcher Maintenance**:
   * If modifying `start-server.bat` or `start.bat`, ensure `--models-dir .\Models` and `--models-max 1` are preserved.
6. **AI Agent as Benchmark Grader**:
   * Always conduct benchmark evaluations directly as the AI agent with 0.01 precision scoring, using `scripts/benchmark_dataset_30.json` as the source dataset. Do not replace agent evaluations with automated regex scripts.

