# Per-Layer NVTX Markers for Diffusion Pipeline Profiling

**Branch:** `misunp/diffusion_02272026`
**Date:** 2026-02-27

---

## Overview

Adds `--enable-layerwise-nvtx-marker` support for the diffusion pipeline (multimodal_gen),
enabling per-layer profiling with Nsight Systems. This covers both the **denoising stage**
(transformer layers) and the **text encoding stage** (text encoder layers), plus
stage-level markers in both sync and parallel executors.

## Comparison with Text LLM (srt) NVTX Implementation

The text LLM path (`python/sglang/srt/`) already has layerwise NVTX profiling via
`srt/utils/nvtx_pytorch_hooks.py`. The diffusion implementation follows the same
patterns but adapts them for the multi-stage diffusion pipeline:

| Aspect | Text LLM (`srt`) | Diffusion (`multimodal_gen`) |
|--------|-------------------|------------------------------|
| **Hook class** | `PytHooks` in `srt/utils/nvtx_pytorch_hooks.py` | Inline methods on `DenoisingStage` and `TextEncodingStage` |
| **CLI flag** | `--enable-layerwise-nvtx-marker` | Same flag (shared `ServerArgs` field) |
| **Registration** | `model_runner.py` at model load | `DenoisingStage.__init__` (eager) / `TextEncodingStage.forward` (lazy) |
| **Marker format** | JSON dict: `{"Module": ..., "TrainableParams": ..., "Inputs": ..., "StaticParams": ...}` | Simple string: `"module_name in=[[shapes]]"` |
| **Module metadata** | Rich — includes trainable params, static params (in_features, etc.) | Lightweight — module name + input shapes only |
| **Hook granularity** | All named_modules (every sub-module) | Block-level only: direct children + ModuleList elements |
| **Scope** | Model forward pass only | Model layers + denoising loop structure + stage-level markers |
| **Additional markers** | None | `denoising_loop`, `denoising_step_i_tXXX`, `predict_noise_cfg`, `scheduler_step`, `stage_*` |

### Why not reuse `PytHooks` from srt?

1. **Import isolation** — `multimodal_gen` and `srt` are separate packages; cross-importing would create undesirable coupling.
2. **Lighter markers** — Diffusion profiling needs less metadata per marker (no need for `TrainableParams`/`StaticParams` dicts). Hooks are registered only at the block level (direct children + ModuleList elements), not on every sub-module. This is critical: `named_modules()` on a 20B-param transformer returns thousands of modules, producing millions of NVTX markers per denoising loop and causing extreme overhead under nsys profiling.
3. **Pipeline-level markers** — The diffusion path needs markers at the denoising loop/step/stage level, which don't exist in the srt path.
4. **Lazy registration** — `TextEncodingStage` doesn't have `server_args` at construction time (it's constructed in pipeline builders without server config), so hooks must be registered lazily on first `forward()` call.

---

## Files Changed

| File | Lines | What |
|------|-------|------|
| `server_args.py` | +11 | Add `enable_layerwise_nvtx_marker` field + CLI argument |
| `stages/denoising.py` | +82 | Hook registration + denoising loop/step/noise/scheduler NVTX markers |
| `stages/text_encoding.py` | +62 | Lazy hook registration + encoder forward NVTX markers |
| `executors/sync_executor.py` | +8 | Stage-level NVTX markers |
| `executors/parallel_executor.py` | +15 | Stage-level NVTX markers for all parallelism types |

**Total:** +178 lines, 0 deletions

---

## NVTX Marker Hierarchy

When profiling with Nsight Systems, you'll see a nested hierarchy:

```
stage_InputValidationStage
stage_TextEncodingStage
  ├── text_encoder_0_forward
  │   ├── text_encoder_0.model                    (inner model — direct child)
  │   ├── text_encoder_0.layers.0                 (decoder layer 0 — ModuleList element)
  │   ├── text_encoder_0.layers.1                 (decoder layer 1)
  │   └── ...                                     (one marker per decoder layer)
stage_LatentPreparationStage
stage_TimestepPreparationStage
stage_DenoisingStage
  └── denoising_loop
      ├── denoising_step_0_t999
      │   ├── predict_noise_cfg
      │   │   ├── transformer.pos_embed           (direct child)
      │   │   ├── transformer.patch_embed         (direct child)
      │   │   ├── transformer.transformer_blocks.0  (DiT block 0 — ModuleList element)
      │   │   ├── transformer.transformer_blocks.1  (DiT block 1)
      │   │   ├── ...                             (one marker per transformer block)
      │   │   ├── transformer.norm_out            (direct child)
      │   │   └── transformer.proj_out            (direct child)
      │   └── scheduler_step
      ├── denoising_step_1_t979
      │   └── ...
      └── ... (×50 steps)
stage_DecodingStage
```

**Hook count:** ~30–40 modules per model (direct children + ModuleList elements)
vs. the previous approach which hooked thousands of sub-modules via `named_modules()`.

---

## Usage

### Known issue: `sglang serve` CLI wrapper

The `sglang` binary is a bash wrapper (`#!/bin/bash` + `python -m sglang.cli.main "$@"`).
For diffusion models, the CLI detects the model type and spawns workers via
`multiprocessing`. The parent process exits immediately, causing:
- The server to appear to exit silently (no output, no error)
- `nsys profile sglang serve ...` to finish instantly (nsys loses the process)

**Workaround:** Use `python -c` to call `main()` directly, keeping everything in one process:

```bash
python -c "
from sglang.cli.main import main
import sys
sys.argv = ['sglang', 'serve', '--model-path', 'Qwen/Qwen-Image-2512', '--port', '8001',
            '--dit-cpu-offload', 'false', '--text-encoder-cpu-offload', 'false',
            '--warmup', '--enable-layerwise-nvtx-marker']
main()
"
```

### Note: `--disable-cuda-graph` is NOT a diffusion flag

The diffusion pipeline (`multimodal_gen.runtime.server_args`) does not have
`--disable-cuda-graph` — that flag only exists in the text LLM path (`srt.server_args`).
The diffusion pipeline doesn't use CUDA graphs.

### With Nsight Systems

```bash
mkdir -p /path/to/output/dir

nsys profile \
  --stats=true \
  -t cuda,nvtx,python-gil,osrt \
  --python-backtrace=cuda \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --sample=process-tree \
  --output=/path/to/output/dir/nsys \
  --force-overwrite=true \
  --wait=all \
  --duration=300 \
  --kill=sigterm \
  python -c "
from sglang.cli.main import main
import sys
sys.argv = ['sglang', 'serve', '--model-path', 'Qwen/Qwen-Image-2512', '--port', '8001',
            '--dit-cpu-offload', 'false', '--text-encoder-cpu-offload', 'false',
            '--warmup', '--enable-layerwise-nvtx-marker']
main()
"
```

Then from another terminal, send requests (e.g. via `aiperf profile` or `curl`).
Open the `.nsys-rep` file in Nsight Systems GUI to see the per-layer timeline.

### Zero overhead when disabled

All NVTX markers are gated by `if use_nvtx:` checks. When `--enable-layerwise-nvtx-marker`
is not set, there is zero runtime overhead — no hooks are registered and no conditionals
are evaluated in hot paths (the flag is checked once and cached in a local variable).

---

## Push/Pop Balance Verification

| Scope | Push location | Pop location |
|-------|--------------|-------------|
| `stage_*` | sync_executor:39 / parallel_executor:81,95,104 | sync_executor:42 / parallel_executor:84,98,107 |
| `denoising_loop` | denoising.py:1081 | denoising.py:1196 |
| `denoising_step_i_tXXX` | denoising.py:1085 | denoising.py:1193 |
| `predict_noise_cfg` | denoising.py:1136 | denoising.py:1154 |
| `scheduler_step` | denoising.py:1162 | denoising.py:1171 |
| `text_encoder_i_forward` | text_encoding.py:323 | text_encoding.py:332 |
| Per-layer hooks (pre/post) | `_nvtx_module_fwd_pre_hook` | `_nvtx_module_fwd_hook` |

All ranges are properly balanced.

---

## Why Block-Level Hooks (not `named_modules()`)

The initial implementation used `named_modules()` to hook every sub-module in the
transformer (every `Linear`, `LayerNorm`, `Attention`, `MLP`, etc.). This is the same
approach used by SRT's `PytHooks` for text LLMs. However, diffusion models have a
critical difference: **the denoising loop runs 50 steps × 2 CFG passes = 100 full
forward passes per image**.

| | Text LLM (SRT) | Diffusion |
|--|-----|------|
| Forward passes per request | 1 per token | **100** (50 steps × 2 CFG) |
| Hooks per pass (named_modules) | ~thousands | ~thousands |
| Total NVTX markers per request | Thousands × N tokens | Thousands × **100** |

With `named_modules()`, this produced **millions of NVTX markers per image**, causing
extreme overhead under nsys profiling. The server would appear to hang during warmup
because each marker has profiling overhead that compounds across all passes.

**Fix:** Use `named_children()` + `ModuleList` expansion to hook only block-level
modules (~30-40 per model). This gives ~3,000-4,000 markers per image — manageable
and still provides per-transformer-block visibility in nsys.

---

## Bug Fix from Previous Implementation

The previous commit (`59adbfe0b`) had an indentation bug where the `nvtx.range_pop()`
for `denoising_step_i` was placed outside the `with StageProfiler` block, causing the
trajectory save, progress bar update, and `step_profile()` to be nested inside the
`if use_nvtx:` conditional — meaning they would only execute when NVTX was enabled.

This version fixes the indentation: the step pop comes after all step logic completes,
outside the `with StageProfiler` block but at the `for` loop level.
