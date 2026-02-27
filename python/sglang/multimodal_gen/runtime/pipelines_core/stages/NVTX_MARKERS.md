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
| **Skip types** | `Identity, Dropout, Dropout1d/2d/3d` | Same |
| **Scope** | Model forward pass only | Model layers + denoising loop structure + stage-level markers |
| **Additional markers** | None | `denoising_loop`, `denoising_step_i_tXXX`, `predict_noise_cfg`, `scheduler_step`, `stage_*` |

### Why not reuse `PytHooks` from srt?

1. **Import isolation** — `multimodal_gen` and `srt` are separate packages; cross-importing would create undesirable coupling.
2. **Lighter markers** — Diffusion profiling needs less metadata per marker (no need for `TrainableParams`/`StaticParams` dicts). The simpler format reduces profiling overhead across 50+ denoising steps × hundreds of layers.
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
  │   ├── text_encoder_0                          (root module)
  │   ├── text_encoder_0.model                    (inner model)
  │   ├── text_encoder_0.model.layers.0           (transformer layer 0)
  │   ├── text_encoder_0.model.layers.0.self_attn
  │   ├── text_encoder_0.model.layers.0.mlp
  │   └── ...
stage_LatentPreparationStage
stage_TimestepPreparationStage
stage_DenoisingStage
  └── denoising_loop
      ├── denoising_step_0_t999
      │   ├── predict_noise_cfg
      │   │   ├── transformer                     (root module)
      │   │   ├── transformer.blocks.0            (DiT block 0)
      │   │   ├── transformer.blocks.0.attn
      │   │   ├── transformer.blocks.0.ff
      │   │   └── ...
      │   └── scheduler_step
      ├── denoising_step_1_t979
      │   ├── predict_noise_cfg
      │   └── scheduler_step
      └── ... (×50 steps)
stage_DecodingStage
```

---

## Usage

### Basic profiling

```bash
sglang serve --model-path Qwen/Qwen-Image-2512 --port 8001 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --enable-layerwise-nvtx-marker
```

### With Nsight Systems

```bash
nsys profile -t cuda,nvtx -o qwen_image_profile \
  sglang serve --model-path Qwen/Qwen-Image-2512 --port 8001 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --enable-layerwise-nvtx-marker
```

Then send a request and open the `.nsys-rep` file in Nsight Systems GUI to see
the per-layer timeline.

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

## Bug Fix from Previous Implementation

The previous commit (`59adbfe0b`) had an indentation bug where the `nvtx.range_pop()`
for `denoising_step_i` was placed outside the `with StageProfiler` block, causing the
trajectory save, progress bar update, and `step_profile()` to be nested inside the
`if use_nvtx:` conditional — meaning they would only execute when NVTX was enabled.

This version fixes the indentation: the step pop comes after all step logic completes,
outside the `with StageProfiler` block but at the `for` loop level.
