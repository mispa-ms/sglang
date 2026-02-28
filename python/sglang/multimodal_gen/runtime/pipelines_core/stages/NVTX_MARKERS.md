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
| **Hook granularity** | All named_modules (every sub-module) | Same — all named_modules (2,477 hooks for Qwen-Image-2512) |
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
  │   ├── text_encoder_0                          (root module)
  │   ├── text_encoder_0.model                    (inner model)
  │   ├── text_encoder_0.model.layers.0           (decoder layer 0)
  │   ├── text_encoder_0.model.layers.0.self_attn (attention)
  │   ├── text_encoder_0.model.layers.0.mlp       (feed-forward)
  │   └── ...                                     (all sub-modules)
stage_LatentPreparationStage
stage_TimestepPreparationStage
stage_DenoisingStage
  └── denoising_loop
      ├── denoising_step_0_t999
      │   ├── predict_noise_cfg
      │   │   ├── transformer                     (root module)
      │   │   ├── transformer.transformer_blocks.0          (DiT block 0)
      │   │   ├── transformer.transformer_blocks.0.attn1    (self-attention)
      │   │   ├── transformer.transformer_blocks.0.attn2    (cross-attention)
      │   │   ├── transformer.transformer_blocks.0.ff       (feed-forward)
      │   │   └── ...                             (all sub-modules, ~2,477 total)
      │   └── scheduler_step
      ├── denoising_step_1_t979
      │   └── ...
      └── ... (×50 steps)
stage_DecodingStage
```

**Hook count:** 2,477 modules for Qwen-Image-2512 (all sub-modules via `named_modules()`).

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

## Hook Granularity

Uses `named_modules()` to hook all sub-modules (Linear, LayerNorm, Attention, MLP, etc.),
matching SRT's `PytHooks` approach. For Qwen-Image-2512 this registers **2,477 hooks**.

Tested and confirmed working both with and without nsys profiling. The total markers per
image generation (~2,477 hooks × 100 forward passes = ~247K markers) is manageable.

**Note:** An earlier investigation suspected this caused server hangs under nsys. This was
disproven — the actual issues were unrelated (see Troubleshooting section below).

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `sglang serve` exits silently, no output | Container's bash wrapper uses `python -m sglang.cli.main` but `main.py` had no `if __name__ == "__main__": main()` guard — Python loaded the module, defined functions, and exited without calling `main()` | Fixed: added `__main__` guard to `cli/main.py`. Also works with `python -c "from sglang.cli.main import main; main()"` |
| nsys profile finishes instantly | Same root cause — `python -m` exits immediately so nsys has nothing to profile | Same fix. Or use `python -c` approach to bypass the bash wrapper |
| `error: unrecognized arguments: --disable-cuda-graph` | Flag only exists in text LLM path (`srt`), not diffusion | Remove it — diffusion doesn't use CUDA graphs |
| `Failed to create ... No such file or directory` | nsys output directory doesn't exist | `mkdir -p /path/to/output/dir` before running |
| Server seems stuck during loading | Model loading takes ~20s, warmup adds ~10s | Wait — the HTTP URL appears after all loading + warmup |
| Ctrl+C doesn't generate nsys trace | `2>&1 \| tee` pipe — SIGINT hits all processes in the pipeline; `tee` exits first causing a broken pipe that kills nsys before it can flush trace data | Run nsys **without** `\| tee`. The `--stats` report and server logs still print to stdout. The `.nsys-rep` file is written on clean shutdown |

---

## Root Cause: `sglang serve` Silent Exit

The container's `/usr/local/bin/sglang` is a bash wrapper:
```bash
#!/bin/bash
python -m sglang.cli.main "$@"
```

`python -m sglang.cli.main` runs `main.py` as `__main__`. But the file only defined
functions — it never called `main()`. Without `if __name__ == "__main__": main()` at
the bottom (and no `sglang/cli/__main__.py`), Python loaded the module and exited
silently. This affected all diffusion and text LLM `sglang serve` calls via this wrapper.

**Why `python -c` worked:** It explicitly calls `main()`:
```python
python -c "from sglang.cli.main import main; main()"
```

**Why pip-installed containers work:** The entry point in `pyproject.toml`
(`sglang = "sglang.cli.main:main"`) generates a script that calls `main()` directly.
Only the hand-written bash wrapper using `python -m` was broken.

**Why this didn't surface before NVTX work:** The bug always existed in `main.py`,
but it was latent. Official SGLang containers (`sglang:dev`, `sglang:latest`) install
via `pip install`, which creates a setuptools entry point that calls `main()` directly —
`python -m` is never used. We started building custom auto_image_builder containers
(to embed `aiperf==0.5.0` and our NVTX branch) at the same time as the NVTX work.
The auto_image_builder creates a bash wrapper using `python -m`, which triggers the bug.
The failure appeared to correlate with NVTX changes, but it was actually caused by
switching from pip-installed entry points to the bash wrapper.

**Fix (commit `81b23d854`):** Added 2 lines to `python/sglang/cli/main.py`:
```python
if __name__ == "__main__":
    main()
```

---

## Bug Fix from Previous Implementation

The previous commit (`59adbfe0b`) had an indentation bug where the `nvtx.range_pop()`
for `denoising_step_i` was placed outside the `with StageProfiler` block, causing the
trajectory save, progress bar update, and `step_profile()` to be nested inside the
`if use_nvtx:` conditional — meaning they would only execute when NVTX was enabled.

This version fixes the indentation: the step pop comes after all step logic completes,
outside the `with StageProfiler` block but at the `for` loop level.
