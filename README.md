# fluxgen-cli

A CLI and Interactive REPL for local AI image generation and instruction-based image editing on macOS using mflux.

- **Generate images** using state-of-the-art `mflux` backends: `zimage-turbo`, `zimage`, `flux2-klein4b`, `flux2-klein9b`, `krea2` (Krea 2 Turbo), and `qwen21` (Qwen-Image-2.1).
- **Edit images** via natural language instructions using `flux2-klein-edit` (multi-image support).
- **Interactive REPL Mode** to keep models cached persistently in memory for near-instant successive runs.
- **Robust Path Security & Validation** protecting against directory traversal and corrupted image inputs.
- **Local Defaults** customizable in a simple `.fluxgen.toml` configuration.

---

## Installation

Install globally using `uv`:

```bash
uv tool install .
```

For local development and testing:

```bash
uv sync --dev
uv run fluxgen --help
```

---

## Requirements & Authentication

- Python 3.10+
- **Apple Silicon Mac** with 32 GB+ unified memory is highly recommended.
- **Hugging Face Hub Access** for model downloads.

### Hugging Face Authentication
The CLI downloads model weights from the Hugging Face Hub on first use. To authenticate:

1. Obtain a User Access Token from your [Hugging Face Security Tokens Settings](https://huggingface.co/docs/hub/en/security-tokens).
2. Log in using the Hugging Face CLI (automatically installed with dependencies):
   ```bash
   huggingface-cli login
   ```
3. Alternatively, export your token as an environment variable in your terminal session:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

> [!NOTE]
> Running the CLI will download models on first use. The `flux2-klein*` and `zimage*` weights are several gigabytes. Ensure you have a stable internet connection and sufficient disk space.

---

## Usage

If no subcommand is provided, `fluxgen` defaults to treating the first argument as a generation prompt.

### 1. Image Generation

```bash
fluxgen "A beautiful cinematic mountain landscape"
fluxgen gen "A bustling cyberpunk city scene" --preset standard
fluxgen gen "A fantasy world" --style cinematic
fluxgen gen "A playful puppy" --model zimage
fluxgen gen "A playful puppy" --model zimage --no-timer
fluxgen gen "Abstract mural" -r 16:9          # widescreen
fluxgen gen "Portrait sketch" --resolution 3:4 # portrait ratio
fluxgen gen "Quick test" --width 800           # override width, keep preset height
```

#### Krea 2 (`--model krea2`)

`krea2` runs [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), an 8-step-distilled
checkpoint (CFG 1.0). Recommended invocation:

```bash
fluxgen gen "A photograph of a red fox in a sunlit forest" --model krea2 --steps 8 -q 8
```

Caveats:

- The shared presets (`fast` 5 / `standard` 9 / `quality` 16 steps) predate this model's
  8-step distillation and are not its sweet spot — pass `--steps 8` (or your own value)
  explicitly for best quality.
- First run downloads ~33 GB of weights (~24 GB transformer + text encoder + VAE).
  Quantize with `-q 8`; budget for ~32 GB+ unified memory.
- `--init-image` uses strength-based img2img (VAE encode + denoise). It is **not** Krea's
  hosted style-reference path, which this CLI does not support.

#### Qwen-Image-2.1 (`--model qwen21`)

`qwen21` runs [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), a 7B single-stream
block-causal DiT with a Qwen3-VL text encoder. It is guidance-free (no guidance value is ever
passed to mflux for this model) and defaults to `--quantize 4` weights. Recommended invocation:

```bash
fluxgen gen "A photograph of a red fox in a sunlit forest" --model qwen21 --steps 40
```

Caveats:

- Requires `mflux >= 0.20.0` (the `qwen21` module landed upstream there).
- The shared presets (`fast` 5 / `standard` 9 / `quality` 16 steps) predate this model's
  recommended sampling and are not its sweet spot — pass `--steps 40` (or your own value)
  explicitly for best quality.
- First run downloads ~33 GB of bf16 weights. The Qwen3-VL text encoder (~17.5 GB bf16) is
  **never quantized** and stays resident regardless of `--quantize`; the default
  `--quantize 4` keeps the transformer at ~4 GB so budget roughly ~24 GB+ of unified memory.
  Pass `--quantize 8` for maximum fidelity (upstream measures unquantized bf16 as the most
  accurate path, peaking around ~46 GB on 64 GB machines).
- `--quantize` always wins over this model's q4 default; the shared presets' quantize values
  are ignored for it.
- True CFG exists upstream but requires a negative prompt, which this CLI does not plumb;
  presets carrying `guidance` are ignored for this model.
- `--init-image` uses strength-based img2img. The upstream multi-reference / RGBA editing
  variant (`QwenImage21Edit`) is **not available yet** — it exists only on mflux's GitHub
  main, unreleased as of 0.20.0, so editing stays on `flux2-klein-edit`.

### 2. Instruction-Based Image Editing

The `edit` command uses `flux2-klein-edit` (MLX). **Supports editing multiple input images at once.**

```bash
# Single image edit
fluxgen edit photo.jpg "add a red hat to the cat"

# Multi-image batch edit
fluxgen edit shot1.png shot2.png shot3.png "make it sunset synthwave style"
```

### 3. Interactive REPL Mode

To completely eliminate the model loading overhead between consecutive generations or edits, start a persistent interactive shell:

```bash
fluxgen interactive
# Or:
fluxgen repl
```

Inside the REPL, models remain warmed up in memory:
```text
fluxgen> gen "A mystical forest" --style cinematic
fluxgen> edit output/generated-file.png "add a castle in the background"
fluxgen> help
```

---

## Command Reference

| Command | Usage | Description |
|---|---|---|
| `generate`, `gen` | `fluxgen gen "prompt"` | Generate an image from text or a reference image |
| `edit` | `fluxgen edit input.png "instruction"` | Edit existing image(s) using natural language instructions |
| `interactive`, `repl` | `fluxgen interactive` | Start an interactive REPL session (keeps models in memory) |

---

## CLI Options

### Global Options
- `-s`, `--silent`: Suppress non-error console output.
- `-v`, `--verbose`: Show debug output and full error tracebacks.

### Generation Options (`generate`, `gen`)
- `--model [zimage-turbo|zimage|flux2-klein4b|flux2-klein9b|krea2|qwen21]`: Model backend to use (default: `zimage-turbo`).
- `-0`, `--fast`: Fast preset (fewer steps).
- `-3`, `--standard`: Standard preset.
- `-8`, `--quality`: Quality preset (higher steps).
- `--preset [fast|standard|quality]`: Named preset override.
- `--style NAME`: Built-in or configured prompt style (default: `none`).
- `--no-style`: Disable styling completely.
- `--steps INT`: Override preset inference steps.
- `--quantize INT`: Override mflux model quantization.
- `--resolution PRESET`, `-r PRESET`: Resolution preset (see table below; default: `tiny` `512x512` for fast generation).
- `--width INT`, `--height INT`: Output dimensions (overrides `--resolution` and config).
- `--output FILE`: Output filename (auto-generated if not specified).
- `--output-dir DIR`: Output directory (default: `output`).
- `--seed INT`: Deterministic random seed.
- `--init-image FILE`: Reference image for image-to-image generation.
- `--strength FLOAT`: Reference image strength (default: `0.4`).
- `--no-timer`: Hide generation time (timer is enabled by default).

#### Resolution Presets

| Preset | Dimensions | Notes |
|---|---|---|
| `tiny` | 512×512 | Default — fast, good for iteration |
| `square` | 768×768 | |
| `large` | 1024×1024 | |
| `full` | 1536×1536 | |
| `1:1` | 512×512 | Square aspect |
| `16:9` | 960×544 | Widescreen |
| `9:16` | 544×960 | Portrait / phone |
| `4:3` | 768×576 | Classic photo |
| `3:4` | 576×768 | Classic portrait |

**Resolution priority** (highest to lowest): `--width`/`--height` flags > `--resolution` flag > `.fluxgen.toml` config > `tiny` default.

### Editing Options (`edit`)
- `--model [flux2-klein-edit]`: Editing model to use (default: `flux2-klein-edit`).
- `--output FILE`: Output filename (default: first input image name appended with a random word).
- `--output-dir DIR`: Output directory (default: `output`).
- `--steps INT`: Inference steps override.
- `--guidance FLOAT`: Guidance scale override.
- `--quantize INT`: Override MLX quantization for `flux2-klein-edit`.
- `--width INT`, `--height INT`: Force specific output dimensions (re-scales while preserving aspect ratio, max 1920px).
- `--seed INT`: Deterministic random seed.
- `--no-timer`: Hide editing time (timer is enabled by default).

---

## Configuration

Create a `.fluxgen.toml` configuration in your user home directory (`~/`) or the current project directory:

```toml
[defaults]
model = "zimage-turbo"
output_dir = "my_images"
width = 1024
height = 1024
preset = 0
style = "none"

[styles]
retro = " in retro 80s style, synthwave colors"
```

A local configuration in the current directory will automatically take precedence over the home directory configuration.

---

## MCP Server (AI Agent Integration)

`fluxgen-mcp` exposes image generation and editing to AI agents over the [Model Context Protocol](https://modelcontextprotocol.io/) (stdio transport). The agent sees two tools:

- `generate_image` — text-to-image with optional image-to-image init. Beyond `prompt`/`model`/`preset`, it accepts per-call `steps` (capped by `max_steps`, overriding the preset), `guidance` (bounded to `[1.0, 20]`; rejected for guidance-free models such as the turbo variants, whose samplers ignore CFG — values below 1.0 are rejected too, since most samplers skip CFG at `<= 1.0` and the caller would be misled about what ran), and `negative_prompt` (subject to the same content filter as the prompt; rejected up front for models that would silently ignore it — Flux.2 Klein lacks the sampler input entirely, the guidance-free turbo variants never encode one, and models whose effective guidance is ≤ 1.0 skip negative encoding, so pass `guidance` > 1.0 alongside it). The result reports the effective `steps` used, and the audit record includes any `negative_prompt` supplied.
- `edit_image` — instruction-based editing (`flux2-klein-edit`, multi-image)

All output paths are sandboxed under `output_root`. Models, dimensions, and prompt lengths are bounded by `.fluxgen.toml`. Every call writes a JSONL audit record (mode `0600`) including the full prompt and a SHA-256 of it. Prompt blocklist is opt-in (default empty).

### Install

```sh
uv pip install "mcp[cli]>=2.0,<3.0" "pydantic>=2.7"
```

### Run

```sh
fluxgen-mcp --print-config     # show resolved settings and exit
fluxgen-mcp                    # stdio server, blocks until killed
```

### Wire into an MCP client

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "fluxgen": {
      "command": "fluxgen-mcp",
      "transport": "stdio"
    }
  }
}
```

### Optional `[mcp]` block in `.fluxgen.toml`

```toml
[mcp]
output_root = "~/fluxgen-mcp-output"
max_width = 1920
max_height = 1920
max_steps = 50
max_prompt_chars = 2000
max_concurrent_jobs = 1
max_queue_depth = 4
per_call_timeout_s = 600
allowed_generation_models = ["zimage-turbo", "zimage", "flux2-klein4b", "flux2-klein9b", "krea2", "qwen21"]
allowed_edit_models = ["flux2-klein-edit"]
prompt_blocklist = []
audit_log_path = "~/.fluxgen-mcp-audit.log"
pause_sentinel_path = "~/.fluxgen-mcp-paused"
pid_file_path = "~/.fluxgen-mcp.pid"
input_max_bytes = 20971520        # 20 MB
input_max_dimension = 1080        # tighten from 1920 output cap
```

### Kill switches

- **Pause:** `touch ~/.fluxgen-mcp-paused` — server rejects new calls with `E_DISABLED` but stays running. Remove the file to resume.
- **Hard stop:** send `SIGTERM` (or `SIGINT`) to the PID in `~/.fluxgen-mcp.pid`.

---

## Project Docs

- [Architecture](ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
