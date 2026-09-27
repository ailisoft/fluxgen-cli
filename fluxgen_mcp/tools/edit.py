"""`edit_image` MCP tool.

Thin wrapper over `fluxgen.editor.ImageEditor.edit` that:
  - validates the prompt against the safety policy,
  - enforces input bounds (size, max dimension),
  - resolves the output path under the sandbox root,
  - delegates the actual edit to `asyncio.to_thread` so the MCP
    event loop stays responsive,
  - attaches seed/model/output_path metadata to `MCPError` so the
    server can build a complete audit record.

Mirrors `fluxgen_mcp.tools.generate` but for the edit pipeline.
Audit logging is owned by the server, so this module does not
write audit records.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from fluxgen.editor import ImageEditor
from fluxgen.generator import generate_random_filename
from fluxgen.models import (
    DEFAULT_EDIT_MODEL,
    EDIT_MODEL_RENAMES,
    REMOVED_EDIT_MODELS,
    SUPPORTED_EDIT_MODELS,
)

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import E_BAD_ARG, E_MODEL, MCPError, EXCEPTION_MAP
from fluxgen_mcp.safety import check_pause, resolve_sandbox_output, validate_prompt
from fluxgen_mcp.validation import (
    resolve_guidance as _resolve_guidance,
    resolve_steps as _resolve_steps,
    validate_edit_inputs,
)

logger = logging.getLogger("fluxgen-mcp")


def _edit_model_error_hint(model_name: str) -> str:
    if model_name in EDIT_MODEL_RENAMES:
        return f" (renamed to {EDIT_MODEL_RENAMES[model_name]!r})"
    if model_name in REMOVED_EDIT_MODELS:
        return f" (removed; use {DEFAULT_EDIT_MODEL!r})"
    return ""


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _edit_filename() -> str:
    """Filename for an edit output.

    Uses `fluxgen.generator.generate_random_filename()` which
    combines a 3-word wonderwords prefix with a millisecond +
    4-hex-char timestamp fallback to guarantee uniqueness even
    under sustained concurrent calls. We deliberately do NOT
    prepend the input stem (the CLI's `handle_edit` does this)
    because the stem + 1-word wonderwords suffix can collide
    under load; the safer path is to trust the CLI generator's
    uniqueness guarantees. Input paths are still preserved in
    the audit log for traceability.
    """
    return generate_random_filename()


def _resolve_dim(d: int | None, settings: MCPSettings, axis: str) -> int | None:
    if d is None:
        return None
    cap = settings.max_width if axis == "width" else settings.max_height
    if d < 1 or d > cap:
        raise MCPError(
            E_BAD_ARG,
            f"{axis} must be in [1, {cap}]; got {d}",
        )
    return int(d)


async def edit_image_tool(
    *,
    settings: MCPSettings,
    input_paths: list[str],
    prompt: str,
    model: str | None,
    seed: int | None,
    steps: int | None,
    guidance: float | None,
    width: int | None,
    height: int | None,
    output_subdir: str,
) -> dict[str, Any]:
    """Run one edit. Returns a dict matching the tool's output schema.

    Raises:
        MCPError: any safety or input validation failure.
        fluxgen.exceptions.FluxgenError / OSError / RuntimeError:
            underlying pipeline errors are caught and re-raised as
            `MCPError(E_MODEL)`.
    """
    check_pause(settings)
    validate_prompt(settings, prompt)

    target_model = model or DEFAULT_EDIT_MODEL
    if target_model not in settings.allowed_edit_models:
        hint = _edit_model_error_hint(target_model)
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} not in allowed_edit_models{hint}",
        )
    if target_model not in SUPPORTED_EDIT_MODELS:
        hint = _edit_model_error_hint(target_model)
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} not supported by fluxgen-cli{hint}",
        )

    if not input_paths:
        raise MCPError(E_BAD_ARG, "input_paths must be non-empty")

    resolved_inputs = validate_edit_inputs(
        input_paths,
        max_bytes=settings.input_max_bytes,
        max_dimension=settings.input_max_dimension,
    )

    final_steps = _resolve_steps(steps, settings)
    final_guidance = _resolve_guidance(guidance)
    final_w = _resolve_dim(width, settings, "width")
    final_h = _resolve_dim(height, settings, "height")
    final_seed = _resolve_seed(seed)

    output_dir = resolve_sandbox_output(settings, output_subdir)
    output_path = str(output_dir / _edit_filename())

    started = time.perf_counter()
    try:
        editor = ImageEditor(model_name=target_model)

        def _run_edit() -> None:
            editor.load()
            editor.edit(
                image_paths=[str(p) for p in resolved_inputs],
                prompt=prompt,
                output_path=output_path,
                steps=final_steps,
                guidance_scale=final_guidance,
                seed=final_seed,
                width=final_w,
                height=final_h,
            )

        await asyncio.to_thread(_run_edit)
    except tuple(EXCEPTION_MAP.keys()) as exc:
        code = EXCEPTION_MAP.get(type(exc), E_MODEL)
        raise MCPError(
            code,
            str(exc),
            seed=final_seed,
            model=target_model,
            output_path=None,
        ) from exc
    except (OSError, RuntimeError) as exc:
        # Model-load failures, MLX OOM, NaN outputs, etc.
        logger.warning(
            "tool edit_image: %s: %s", type(exc).__name__, exc,
        )
        raise MCPError(
            E_MODEL,
            f"{type(exc).__name__}: {exc}",
            seed=final_seed,
            model=target_model,
            output_path=None,
        ) from exc

    elapsed = time.perf_counter() - started
    return {
        "path": output_path,
        "elapsed_s": round(elapsed, 3),
        "model": target_model,
        # Audit traceability: include the resolved input paths so
        # the audit log captures both what the tool acted on and
        # what it produced. Stored as resolved `Path` strings.
        "input_paths": [str(p) for p in resolved_inputs],
    }
