# Changelog

## Unreleased

- `qwen21` now defaults to 4-bit weights (`default_quantize=4` on the model spec). Quantization
  resolution is explicit `--quantize` → per-model default → preset → bf16; a model default
  overrides the preset value because presets encode choices made before the model existed,
  while the spec default encodes its memory profile (the ~17.5 GB bf16 text encoder is never
  quantized, so q4 keeps total peak around ~24 GB instead of ~46 GB). Wired through the CLI
  preload, the model-less `generate_image` path (MCP), and `ImageEditor.load`.
- Clarified `qwen21` edit status: upstream mflux 0.20.0 ships **txt2img only** — the
  `QwenImage21Edit` multi-reference/RGBA variant exists only on GitHub main and is unreleased,
  so editing remains on `flux2-klein-edit` (tracked in ROADMAP).
- Added `qwen21` (Qwen-Image-2.1) as a generation-only model: 7B single-stream block-causal
  DiT with a Qwen3-VL text encoder, 40-step guidance-free sampling (CFG 1.0). ~33 GB first-run
  download. Presets carrying `guidance` are ignored for this model (true CFG upstream needs a
  negative prompt, which fluxgen does not plumb).
- Bumped `mflux` floor to `>=0.20.0` (Qwen-Image-2.1 support landed upstream there; 0.19.x
  ships only the older Qwen-Image stack).
- Added `krea2` (Krea 2 Turbo) as a generation-only model: 8-step-distilled, CFG 1.0,
  ~33 GB first-run download (~32 GB+ unified memory recommended). Use `--steps 8 -q 8`;
  the shared presets (5/9/16 steps) predate its distillation and are not its sweet spot.
  Strength-based img2img works via `--init-image` (not Krea's hosted style-reference path).
- Bumped `mflux` floor to `>=0.19.1` (Krea 2 landed in 0.18.1; 0.19.1 adds security
  dependency floors).

## 0.4.0 - 2026-08-07

- Removed `qwen-image-edit` (Diffusers/GGUF/torch stack). Editing is mflux-only.
- Renamed edit model id from `flux2-klein` to `flux2-klein-edit` (no alias).
- Dropped unused deps: `torch`, `diffusers`, `transformers`, `accelerate`, `torchvision`, `gguf`.
- Consolidated model IDs / defaults / factories into `fluxgen/models.py`.
- Fixed guidance defaults: `Preset.guidance = None` no longer shadows per-model guidance.
- Removed Qwen-only `--true-cfg-scale` CLI flag.

## 0.3.3 - 2026-07-22

- Bumped `mflux` floor to `>=0.18.0` (FLUX.2-klein-9b-kv KV-cache speedup on multi-ref edits, `text_encoder_2` crash fix).
- Raised dep floors: `torch>=2.13.0`, `torchvision>=0.28.0`, `diffusers>=0.39.0`, `gguf>=0.19.0`, `pytest>=9.1.1`.
- Transitive bumps: `transformers`, `accelerate`, `safetensors`, `numpy`, `setuptools`, `pillow`.

## 0.3.2 - 2026-06-27

- Replaced `--timer` with `--no-timer` on `generate` and `edit`: timer is enabled by default, pass `--no-timer` to suppress it (opt-out pattern is more intuitive).
- Fixed `--resolution` flag priority: CLI now correctly overrides config file.
- Fixed partial `--width`/`--height` fallthrough: when only one dimension is passed, the
  other falls back to config value before the `tiny` default.
- Refactored `ModelManager` dispatch from else-fallback chain to explicit lookup table.
- Lazy-loaded `torch` import to speed up CLI startup.
- Removed redundant `StyleManager` dict copy and `ModelManager` cache lookup in hot path.
- Added resolution presets: `tiny`, `square`, `large`, `full` + aspect ratios `1:1`, `4:3`,
  `3:4`, `16:9`, `9:16`.

## 0.3.0 - 2026-05-16

- Removed `flux1-schnell` model from supported backends (deprecated).
- Split `flux2-klein` into `flux2-klein4b` (4B, default) and `flux2-klein9b` (9B). `flux2-klein` identifier removed.

## 0.2.0 - 2026-05-12

- Changed edit default output names to random Wonderwords filenames.
- Added Qwen-Image-Edit 2511 editing through Diffusers with GGUF transformer weights.
- Added `fluxgen edit image prompt` command for instruction-based image editing.
- Added accelerator-aware edit dtype handling: `bfloat16` on MPS/CUDA, `float32` on CPU.
- Added invalid edit output detection so NaN/blank black results fail instead of being saved.
- Added editor tests for pipeline wiring, defaults, dtype selection, and blank-output rejection.
- Refreshed README and added architecture documentation.
- Added `flux2-klein` (FLUX.2 Klein 9B) to supported models.

## 0.1.6

- Added initial `edit` command using Qwen-Image-Edit for instruction-based image editing.
- Refactored CLI to support subcommands with backward-compatible generation prompts.
- Added `fluxgen/editor.py` with MPS and CUDA device selection.
- Added dependencies: `torch`, `diffusers`, `transformers`, `accelerate`, `torchvision`, and `gguf`.

## 0.1.5

- Changed `--style` default to `none`.
- Added 8 built-in styles.
- Refactored `StyleManager`.

## 0.1.4

- Added `--timer` flag for generation timing.

## 0.1.3

- Added multi-model support through `--model`: `zimage-turbo`, `zimage`, and `flux1-schnell`.
