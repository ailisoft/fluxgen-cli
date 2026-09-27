# Architecture

`fluxgen-cli` is a small Python CLI with two image workflows, both
backed by `mflux` on Apple Silicon (MLX):

- generation (`zimage*`, `flux2-klein*`, `krea2`, `qwen21`)
- instruction editing (`flux2-klein-edit`)

## Entry Point

`fluxgen.cli:main` is the console script declared in `pyproject.toml`.

The CLI has two subcommands:

- `generate` / `gen`: text-to-image and image-to-image generation
- `edit`: instruction-based image editing

For backward compatibility, any first argument that is not a known subcommand is treated as a generation prompt.

## Model registry

All model IDs, capabilities, and default inference params live in
`fluxgen/models.py`. CLI choices, MCP allowlists, and inference
helpers derive from that single registry. Quantization resolves as
explicit `--quantize` → per-model `default_quantize` → preset value →
bf16 (`resolve_quantize`).

## Generation Flow

Files:

- `fluxgen/cli/__init__.py` — entry point (`main`), parser aggregator (`get_parser`), version resolver, logging setup
- `fluxgen/cli/commands.py` — `generate` / `edit` subparsers and handlers, output-path resolver, resolution priority chain (`resolve_image_dimensions`)
- `fluxgen/cli/presets_arg.py` — argv-shape constants, `with_default_command`, and the reusable argparse builders (`add_verbosity_flags`, `add_preset_args`, `add_resolution_args`)
- `fluxgen/cli/interactive.py` — REPL parser subclass and `handle_interactive`
- `fluxgen/models.py`
- `fluxgen/generator.py`
- `fluxgen/presets.py`
- `fluxgen/styling.py`
- `fluxgen/config.py`

Flow:

1. `cli/__init__.py:main` parses argv (after `with_default_command` normalization) and dispatches by subcommand.
2. `config.py` loads `.fluxgen.toml` from home and current directory.
3. `presets.py` supplies step and quantization presets.
4. `cli/commands.py:resolve_image_dimensions` resolves `(width, height)` from CLI flags + config + defaults via the documented priority chain.
5. `styling.py` applies built-in or configured prompt styles.
6. `generator.py` selects an `mflux` model through `ModelManager`.
7. The generated PIL image is saved to the requested output path.

`ModelManager` caches one active model instance. It recreates the model when the requested backend or quantization changes.

Supported generation backends:

- `zimage-turbo` (default)
- `zimage`
- `flux2-klein4b`
- `flux2-klein9b`
- `krea2` (Krea 2 Turbo; 8-step distilled, ~33 GB download, ~32 GB+ unified memory)
- `qwen21` (Qwen-Image-2.1; 40-step guidance-free, ~33 GB download, ~46 GB peak —
  needs `mflux >= 0.20.0`)

`krea2` is generation-only — there is no mflux edit checkpoint, so it is not
registered under `fluxgen edit`. `qwen21` is generation-only too: upstream
ships a `QwenImage21Edit` variant, but its signature (multi-`image_paths`,
`output_resolution`, no `image_strength`) does not fit the editor contract.

## Editing Flow

Files:

- `fluxgen/cli/__init__.py`
- `fluxgen/cli/commands.py`
- `fluxgen/editor.py`
- `fluxgen/models.py`

Flow:

1. `cli/commands.py:handle_edit` builds the edit output path (via `resolve_output_path`) and creates `ImageEditor`.
2. `ImageEditor` loads `flux2-klein-edit` through `ModelManager` (MLX).
3. Input images are validated, optionally downscaled to `max_edit_dimension`, and passed to the edit model.
4. The resulting PIL image is saved.

Supported edit backend:

- `flux2-klein-edit` (default; multi-image)

## Configuration

Config filename: `.fluxgen.toml`

Load order:

1. home directory
2. current working directory

Current-directory values override home values. The config supports:

- `[defaults]` for CLI defaults
- `[styles]` for custom prompt style suffixes

## Output

Generation defaults to a random three-word PNG filename in the configured output directory.

Editing defaults to a random three-word PNG filename in the configured output directory.

Both flows create output directories as needed.

## Tests

Tests live in `tests/`.

Run:

```bash
.venv/bin/python -m pytest -q
```

The editor tests mock model loading and verify command wiring, defaults, and input validation.
