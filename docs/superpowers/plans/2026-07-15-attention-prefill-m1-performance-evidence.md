# Attention Prefill M1 Performance Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible A100 performance evidence chain comparing only `Br=1` and `Br=4` FP32 SIMT Attention.

**Architecture:** Add a dedicated evidence runner that links the frozen FlashAttention tiled library and the M1 Query-tiled library, sharing input generation, validation, and CUDA Event timing. Surround it with tested benchmark rendering, ncu profiling, SASS extraction, and result documentation; do not modify either kernel body.

**Tech Stack:** CUDA C++17, CMake/CTest, Bash, Python 3, CUDA Events, Nsight Compute, `cuobjdump`.

---

### Task 1: Unified Evidence Runner

**Files:**
- Create: `projects/attention_prefill/evidence/main.cu`
- Create: `projects/attention_prefill/evidence/runner.hpp`
- Create: `projects/attention_prefill/evidence/runner.cu`
- Create: `projects/attention_prefill/tests/evidence_runner_tests.cpp`
- Modify: `projects/attention_prefill/CMakeLists.txt`

- [ ] Add failing parser and contract tests for `--implementation br1|br4`, `--mode validate|benchmark`, warmup, iterations, seed, `N/D/causal`, and stable output fields.
- [ ] Build and run the test to verify RED because evidence runner support does not exist.
- [ ] Implement shared deterministic inputs, CPU-double reference, validation, dispatch to both existing launchers, and CUDA Event kernel-only timing.
- [ ] Link both kernel libraries into one executable without changing their bodies.
- [ ] Run evidence runner tests and representative validation/benchmark smoke cases to verify GREEN.

### Task 2: Canonical Benchmark and Renderer

**Files:**
- Create: `projects/attention_prefill/scripts/benchmark_m1.sh`
- Create: `projects/attention_prefill/scripts/render_m1_results.py`
- Create: `projects/attention_prefill/tests/render_m1_results_test.py`
- Create: `projects/attention_prefill/results/README.md`
- Modify: `.gitignore`
- Modify: `projects/attention_prefill/CMakeLists.txt`

- [ ] Add failing renderer tests for 40-row completeness, speedup/delta/spread, classification, CTA counts, requested K/V elements, and malformed CSV rejection.
- [ ] Run the renderer test to verify RED.
- [ ] Implement canonical benchmark provenance, clean-tree guard, 3-repeat median/min/max/spread, smoke output isolation, and automatic Markdown rendering.
- [ ] Add ignored raw/generated/profile/SASS paths while allowing reviewed artifacts to be force-added.
- [ ] Run renderer tests and a tiny smoke benchmark to verify GREEN.

### Task 3: Nsight Compute and SASS Reproduction

**Files:**
- Create: `projects/attention_prefill/scripts/profile_m1.sh`
- Create: `projects/attention_prefill/scripts/extract_m1_sass.sh`
- Create: `projects/attention_prefill/scripts/summarize_m1_ncu.py`
- Create: `projects/attention_prefill/tests/summarize_m1_ncu_test.py`
- Modify: `projects/attention_prefill/CMakeLists.txt`

- [ ] Add failing summary tests for resource, occupancy, eligible-warp, scoreboard, barrier, cache/DRAM, and implementation identity fields.
- [ ] Run tests to verify RED.
- [ ] Implement six-report ncu collection with exact demangled kernel filters and compact CSV/Markdown summarization.
- [ ] Implement SASS extraction for both kernel symbols and static opcode counts (`FFMA`, barrier, global/shared, local, `HMMA`, `LDGSTS`).
- [ ] Run summary tests, one ncu smoke profile per implementation, and SASS extraction to verify GREEN.

### Task 4: Collect and Publish M1 Evidence

**Files:**
- Create: `projects/attention_prefill/results/raw/a100-fp32-m1.csv`
- Create: `projects/attention_prefill/results/generated/a100-fp32-m1.md`
- Create: `projects/attention_prefill/results/evidence/m1-ncu-summary.md`
- Create: `projects/attention_prefill/results/evidence/m1-sass.md`
- Modify: `projects/attention_prefill/README.md`
- Modify: `projects/attention_prefill/ROADMAP.md`

- [ ] Fresh-build the evidence runner and rerun correctness/sanitizer gates.
- [ ] Run the 40-row canonical CUDA Event benchmark on a clean commit and retain provenance.
- [ ] Collect six ncu reports and render the compact summary; do not use ncu duration as wall-clock evidence.
- [ ] Extract SASS and render compact static evidence.
- [ ] Write the final interpretation tying latency to requested loads, measured cache/DRAM behavior, stalls, eligible warps, occupancy, and barriers.
- [ ] Run all project CTests, full sanitizer, Markdown validation, script syntax checks, and `git diff --check` before claiming completion.
