"""`generate_image` MCP tool.

Thin wrapper over `fluxgen.generator.generate_image` that:
  - validates the prompt and negative prompt against the safety policy,
  - enforces model + dimension + preset + steps bounds,
  - resolves the output path under the sandbox root,
  - delegates the actual generation to `asyncio.to_thread` so the
    MCP event loop stays responsive,
  - attaches seed/model/output_path metadata to `MCPError` so the
    server can build a complete audit record.

The model loading / generation logic is reused from `fluxgen`
unchanged — we do not fork the inference code. Audit logging is
owned by the server (`server._with_safety`), so this module does
not write audit records.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict
from typing import Any

from fluxgen.generator import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    generate_image,
    generate_random_filename,
)
from fluxgen.models import MODELS, get_model_spec
from fluxgen.presets import PRESETS, PRESETS_BY_NAME

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BAD_ARG,
    E_MODEL,
    E_PROMPT_REJECTED,
    MCPError,
    EXCEPTION_MAP,
)
from fluxgen_mcp.safety import check_pause, resolve_sandbox_output, validate_prompt
from fluxgen_mcp.validation import (
    resolve_guidance,
    resolve_steps,
    validate_init_image,
)

logger = logging.getLogger("fluxgen-mcp")


_VALID_PRESETS = tuple(PRESETS_BY_NAME.keys())


def _coerce_strength(strength: float | None) -> float:
    if strength is None:
        return 0.4
    if not 0.0 <= strength <= 1.0:
        raise MCPError(E_BAD_ARG, f"strength must be in [0.0, 1.0]; got {strength}")
    return float(strength)


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _resolve_preset(preset: str | None) -> dict[str, Any]:
    name = preset or "fast"
    if name not in _VALID_PRESETS:
        raise MCPError(
            E_BAD_ARG,
            f"preset must be one of {_VALID_PRESETS}; got {preset!r}",
        )
    p = PRESETS[PRESETS_BY_NAME[name]]
    return asdict(p)


async def generate_image_tool(
    *,
    settings: MCPSettings,
    prompt: str,
    model: str | None,
    preset: str | None,
    steps: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    width: int | None,
    height: int | None,
    seed: int | None,
    style: str | None,
    init_image_path: str | None,
    strength: float | None,
    output_subdir: str,
) -> dict[str, Any]:
    """Run one generation. Returns a dict matching the tool's output schema.

    Raises:
        MCPError: any safety or input validation failure (codes in
            `errors.py`). The `seed`, `model`, and `output_path`
            instance attributes are populated whenever known so the
            server's audit writer can include them.
        fluxgen.exceptions.FluxgenError / OSError / RuntimeError:
            underlying model errors are caught and re-raised as
            `MCPError(E_MODEL)`.
    """
    check_pause(settings)
    validate_prompt(settings, prompt)

    # Model whitelist — the MCP server exposes only the deployment-approved
    # list. The default derives from the CLI registry (so ``krea2`` is allowed
    # unless a deployment pins ``allowed_generation_models`` in ``.fluxgen.toml``);
    # deployments that want to keep large first-run downloads off the MCP
    # surface can drop it from that list.
    target_model = model or DEFAULT_MODEL
    if target_model not in settings.allowed_generation_models:
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} not in allowed_generation_models",
        )
    # Defensive: a misconfigured whitelist may name a model the CLI
    # does not support. Surface as a bad-argument error rather than
    # letting it blow up later inside `ModelManager.get_model`.
    if target_model not in SUPPORTED_MODELS:
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} not supported by fluxgen-cli",
        )

    preset_dict = _resolve_preset(preset)
    # Explicit caller steps/guidance override the preset value; both
    # are capped by the same deployment settings. resolve_steps rejects
    # out-of-range values; the preset branch keeps its historical
    # "exceeds max_steps" wording.
    final_steps = resolve_steps(steps, settings)
    if final_steps is None and preset_dict["steps"] > settings.max_steps:
        raise MCPError(
            E_BAD_ARG,
            f"preset steps {preset_dict['steps']} exceeds max_steps {settings.max_steps}",
        )
    final_guidance = resolve_guidance(guidance)

    # Model-level parameter compatibility, rejected here (E_BAD_ARG)
    # rather than surfacing as a model failure or, worse, being
    # silently ignored by mflux:
    #   - guidance-free models (turbo variants) would coerce guidance
    #     to 0.0, making a caller believe CFG applied when it did not;
    #   - models whose generate_image signature lacks negative_prompt
    #     (Flux.2 Klein) would raise a bare TypeError inside mflux.
    target_spec = get_model_spec(target_model)
    if final_guidance is not None and target_spec.guidance is None:
        raise MCPError(
            E_BAD_ARG,
            f"model {target_model!r} is guidance-free; it does not accept guidance",
        )

    # Negative conditioning shares the prompt safety policy (length cap
    # + blocklist). An empty string is a valid explicit value, so only
    # None means "not requested".
    final_negative_prompt: str | None = None
    if negative_prompt is not None:
        if not isinstance(negative_prompt, str):
            raise MCPError(E_BAD_ARG, "negative_prompt must be a string")
        try:
            validate_prompt(settings, negative_prompt)
        except MCPError as exc:
            # Same code and policy as the prompt filter, but the message
            # must not claim the (clean) prompt was rejected.
            if exc.code == E_PROMPT_REJECTED:
                raise MCPError(
                    E_PROMPT_REJECTED, "negative_prompt rejected by content filter"
                ) from exc
            raise
        if not target_spec.supports_negative_prompt:
            raise MCPError(
                E_BAD_ARG,
                f"model {target_model!r} does not support negative_prompt; "
                f"models with support: "
                f"{', '.join(n for n, s in MODELS.items() if s.supports_negative_prompt)}",
            )
        # Even on supported models, negative encoding only runs when CFG
        # is active: mflux skips it at guidance <= 1.0 (z-image) / at
        # guidance == 1.0 (krea2, whose spec default is 1.0). Sub-1.0
        # guidance is already rejected by resolve_guidance, so this guard
        # only ever fires at exactly 1.0. Effective guidance mirrors the
        # generator's resolution: explicit beats spec.
        effective_guidance = (
            final_guidance if final_guidance is not None else target_spec.guidance
        )
        if effective_guidance is None or effective_guidance <= 1.0:
            raise MCPError(
                E_BAD_ARG,
                f"negative_prompt requires guidance > 1.0 on {target_model!r} "
                f"(effective guidance would be {effective_guidance}); "
                f"pass guidance > 1.0 or drop negative_prompt",
            )
        final_negative_prompt = negative_prompt

    final_w = width if width is not None else 512
    final_h = height if height is not None else 512
    if final_w > settings.max_width:
        raise MCPError(
            E_BAD_ARG,
            f"width {final_w} exceeds max_width {settings.max_width}",
        )
    if final_h > settings.max_height:
        raise MCPError(
            E_BAD_ARG,
            f"height {final_h} exceeds max_height {settings.max_height}",
        )

    final_strength = _coerce_strength(strength)
    final_seed = _resolve_seed(seed)

    # Init image (optional). Validated for size, dimensions, integrity.
    resolved_init: str | None = None
    if init_image_path:
        resolved_init = str(
            validate_init_image(
                init_image_path,
                max_bytes=settings.input_max_bytes,
                max_dimension=settings.input_max_dimension,
            )
        )

    output_dir = resolve_sandbox_output(settings, output_subdir)
    output_path = str(output_dir / generate_random_filename())

    started = time.perf_counter()
    try:
        await asyncio.to_thread(
            generate_image,
            prompt=prompt,
            preset=preset_dict,
            steps=final_steps,
            guidance=final_guidance,
            negative_prompt=final_negative_prompt,
            seed=final_seed,
            output=output_path,
            width=final_w,
            height=final_h,
            style=style or "none",
            init_image=resolved_init,
            strength=final_strength,
            model_name=target_model,
        )
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
        # Catch-all for model-side failures that EXCEPTION_MAP does
        # not cover (MLX OOM, etc.). Map to E_MODEL so the agent
        # sees a clear category.
        logger.warning(
            "tool generate_image: %s: %s", type(exc).__name__, exc,
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
        "width": final_w,
        "height": final_h,
        "seed": final_seed,
        # Effective step count actually used (explicit override or the
        # preset the caller selected) so agents can correlate quality
        # with sampling budget.
        "steps": final_steps if final_steps is not None else preset_dict["steps"],
        "elapsed_s": round(elapsed, 3),
        "model": target_model,
    }