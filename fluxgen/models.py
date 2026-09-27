"""Single source of truth for supported mflux models.

Generation and edit backends are registered here. CLI, MCP, and
inference helpers all derive allowlists / defaults from this module
so model IDs cannot drift across entry points.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

Capability = Literal["generate", "edit"]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one backed model."""

    name: str
    capabilities: frozenset[Capability]
    steps: int
    # ``None`` means the model does not accept a guidance kwarg
    # (e.g. guidance-free turbo variants).
    guidance: float | None
    factory: Callable[[int | None], Any]
    # Quantization applied when neither the CLI ``--quantize`` flag nor
    # a higher-priority source names one. ``None`` keeps the historical
    # behavior (preset quantize wins, else bf16).
    default_quantize: int | None = None
    # Whether the model's ``generate_image`` accepts a
    # ``negative_prompt`` kwarg (true-CFG / negative-conditioning
    # models). Flux.2 Klein's signature lacks it, so callers must
    # not pass one there.
    supports_negative_prompt: bool = False


def _make_zimage(quantize: int | None):
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image import ZImage

    return ZImage(quantize=quantize, model_config=ModelConfig.z_image())


def _make_zimage_turbo(quantize: int | None):
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image import ZImageTurbo

    return ZImageTurbo(quantize=quantize, model_config=ModelConfig.z_image_turbo())


def _make_flux2_klein4b(quantize: int | None):
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    return Flux2Klein(quantize=quantize, model_config=ModelConfig.flux2_klein_4b())


def _make_flux2_klein9b(quantize: int | None):
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    return Flux2Klein(quantize=quantize, model_config=ModelConfig.flux2_klein_9b())


def _make_flux2_klein_edit(quantize: int | None):
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    return Flux2KleinEdit(quantize=quantize, model_config=ModelConfig.flux2_klein_9b())


def _make_krea2(quantize: int | None):
    # Krea 2 Turbo — an 8-step-distilled single-stream MMDiT on the
    # Qwen-Image stack. Requires mflux >= 0.19.1 (Krea 2 landed upstream
    # in 0.18.1; 0.19.1 adds the security dependency floors).
    from mflux.models.common.config import ModelConfig
    from mflux.models.krea2 import Krea2

    return Krea2(quantize=quantize, model_config=ModelConfig.krea2())


def _make_qwen21(quantize: int | None):
    # Qwen-Image-2.1 — 7B unified DiT with a Qwen3-VL text encoder.
    # Requires mflux >= 0.20.0 (the qwen21 module landed upstream there);
    # 0.19.x ships only the older Qwen-Image stack.
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    return QwenImage21(quantize=quantize, model_config=ModelConfig.qwen_image_21())


_GENERATE = frozenset({"generate"})
_EDIT = frozenset({"edit"})

MODELS: dict[str, ModelSpec] = {
    # z-image-turbo runs guidance-free (mflux coerces guidance to 0.0 and
    # skips negative encoding entirely), so negative_prompt would be a
    # guaranteed silent no-op — flagged unsupported despite the mflux
    # signature accepting the kwarg.
    "zimage-turbo": ModelSpec(
        name="zimage-turbo",
        capabilities=_GENERATE,
        steps=4,
        guidance=None,
        supports_negative_prompt=False,
        factory=_make_zimage_turbo,
    ),
    "zimage": ModelSpec(
        name="zimage",
        capabilities=_GENERATE,
        steps=20,
        guidance=4.0,
        supports_negative_prompt=True,
        factory=_make_zimage,
    ),
    "flux2-klein4b": ModelSpec(
        name="flux2-klein4b",
        capabilities=_GENERATE,
        steps=4,
        guidance=3.5,
        factory=_make_flux2_klein4b,
    ),
    "flux2-klein9b": ModelSpec(
        name="flux2-klein9b",
        capabilities=_GENERATE,
        steps=4,
        guidance=3.5,
        factory=_make_flux2_klein9b,
    ),
    # Krea 2 Turbo: timestep-distilled to 8 steps (CFG 1.0). Generation-only;
    # there is no mflux edit checkpoint, so it must not appear under ``edit``.
    # Recommend ``--steps 8`` — the shared presets (5/9/16) predate its
    # distillation and are not its sweet spot. negative_prompt engages only
    # when guidance differs from 1.0 (mflux skips it at the 1.0 default),
    # so the MCP layer requires guidance > 1.0 alongside it.
    "krea2": ModelSpec(
        name="krea2",
        capabilities=_GENERATE,
        steps=8,
        guidance=1.0,
        supports_negative_prompt=True,
        factory=_make_krea2,
    ),
    "flux2-klein-edit": ModelSpec(
        name="flux2-klein-edit",
        capabilities=_EDIT,
        steps=4,
        guidance=1.0,
        factory=_make_flux2_klein_edit,
    ),
    # Qwen-Image-2.1: guidance-free by default (40 steps, CFG 1.0). True CFG
    # exists upstream but needs a negative prompt fluxgen does not plumb, so
    # ``guidance=None`` hard-disables it (same treatment as zimage-turbo).
    # Generation-only for now: the upstream QwenImage21Edit signature
    # (multi-image_paths, output_resolution, no image_strength) does not fit
    # the editor contract. Defaults to 4-bit weights: the bf16 text encoder
    # (~17.5 GB) stays resident either way, and q4 keeps the 7B transformer
    # small enough for sub-64 GB machines.
    "qwen21": ModelSpec(
        name="qwen21",
        capabilities=_GENERATE,
        steps=40,
        guidance=None,
        default_quantize=4,
        factory=_make_qwen21,
    ),
}

DEFAULT_GENERATION_MODEL = "zimage-turbo"
DEFAULT_EDIT_MODEL = "flux2-klein-edit"

SUPPORTED_GENERATION_MODELS: tuple[str, ...] = tuple(
    name for name, spec in MODELS.items() if "generate" in spec.capabilities
)
SUPPORTED_EDIT_MODELS: tuple[str, ...] = tuple(
    name for name, spec in MODELS.items() if "edit" in spec.capabilities
)

# Stale ids that may still appear in older `.fluxgen.toml` / MCP configs.
# Remap or drop at config-load time; CLI has no aliases (argparse rejects).
EDIT_MODEL_RENAMES: dict[str, str] = {
    "flux2-klein": "flux2-klein-edit",
}
REMOVED_EDIT_MODELS: frozenset[str] = frozenset({"qwen-image-edit"})

# Backward-compatible aliases used by existing CLI / MCP import sites.
DEFAULT_MODEL = DEFAULT_GENERATION_MODEL
SUPPORTED_MODELS = list(SUPPORTED_GENERATION_MODELS)


def get_model_spec(model_name: str) -> ModelSpec:
    """Look up a model by id (case-insensitive).

    Raises:
        ValueError: unknown model id.
    """
    key = model_name.lower()
    try:
        return MODELS[key]
    except KeyError as exc:
        known = ", ".join(MODELS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {known}"
        ) from exc


def require_capability(model_name: str, capability: Capability) -> ModelSpec:
    """Return the spec, ensuring it supports ``capability``."""
    spec = get_model_spec(model_name)
    if capability not in spec.capabilities:
        allowed = ", ".join(
            name for name, s in MODELS.items() if capability in s.capabilities
        )
        raise ValueError(
            f"Model '{spec.name}' does not support {capability}. "
            f"Choose from: {allowed}"
        )
    return spec


def resolve_inference_params(
    spec: ModelSpec,
    *,
    steps: int | None = None,
    guidance: float | None = None,
    preset: dict[str, Any] | None = None,
) -> tuple[int, float | None]:
    """Resolve steps / guidance with ``None`` treated as missing.

    Priority (highest to lowest): explicit kwargs → ``preset`` values
    (when the key is present and not ``None``) → ``ModelSpec`` defaults.

    One exception to the plain priority chain: a guidance-free spec
    (``guidance=None``, e.g. turbo variants) ignores *all* guidance —
    preset values and explicit kwargs alike. mflux silently coerces
    guidance to 0.0 for such models, so honoring the kwarg would make
    a caller believe CFG applied when it did not.

    ``Preset`` dataclasses always serialize ``guidance: None``, so a
    plain ``dict.get("guidance", default)`` would incorrectly skip
    model defaults. This helper fixes that.
    """
    preset = preset or {}

    resolved_steps = steps
    if resolved_steps is None:
        resolved_steps = preset.get("steps")
    if resolved_steps is None:
        resolved_steps = spec.steps

    if spec.guidance is None:
        return resolved_steps, None

    resolved_guidance = guidance
    if resolved_guidance is None:
        resolved_guidance = preset.get("guidance")
    if resolved_guidance is None:
        resolved_guidance = spec.guidance

    return resolved_steps, resolved_guidance


def resolve_quantize(
    spec: ModelSpec,
    preset: dict[str, Any] | None = None,
    cli_quantize: int | None = None,
) -> int | None:
    """Resolve quantization: explicit flag → model default → preset → bf16.

    Unlike steps/guidance, a model's ``default_quantize`` overrides the
    preset value: presets carry quantize choices made before the model
    existed (e.g. the shared 5/9/16-step presets), while the spec
    default encodes this model's memory profile (qwen21 at bf16/q8
    peaks ~46 GB because the text encoder is never quantized).
    Explicit CLI ``--quantize`` always wins.
    """
    if cli_quantize is not None:
        return cli_quantize
    if spec.default_quantize is not None:
        return spec.default_quantize
    return (preset or {}).get("quantize")


class ModelManager:
    """Caches one active model instance; recreates on config change."""

    _instance = None
    _current_config = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str, quantize: int | None = None):
        """Return a cached model instance, re-creating only when config changes."""
        spec = get_model_spec(model_name)
        config_key = (spec.name, quantize)
        with cls._lock:
            if cls._instance is None or cls._current_config != config_key:
                cls._instance = spec.factory(quantize)
                cls._current_config = config_key
        return cls._instance

    @classmethod
    def reset(cls):
        """Clear the cached model (useful for switching models / tests)."""
        with cls._lock:
            cls._instance = None
            cls._current_config = None
