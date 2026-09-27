import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from fluxgen.models import (
    DEFAULT_MODEL,
    MODELS,
    ModelManager,
    SUPPORTED_MODELS,
    require_capability,
    resolve_inference_params,
)
from fluxgen.styling import StyleManager

logger = logging.getLogger("fluxgen")

# Re-export registry symbols so existing ``from fluxgen.generator import …``
# call sites (CLI, MCP, tests) keep working.
__all__ = [
    "DEFAULT_MODEL",
    "SUPPORTED_MODELS",
    "ModelManager",
    "generate_image",
    "generate_random_filename",
]


def generate_random_filename() -> str:
    """Generate a random 3-word filename with .png extension.

    Uses wonderwords for the happy path. Falls back to a millisecond
    timestamp + 4-hex-char random suffix if wonderwords is unavailable or
    fails. The suffix guarantees uniqueness for two back-to-back calls
    within the same millisecond (vanishingly rare but possible on fast
    hardware or under monotonic-clock skew).
    """
    if _random_word is not None:
        try:
            words = _random_word.random_words(3, word_max_length=5)
            return "-".join(words) + ".png"
        except Exception as e:
            logger.info(f"wonderwords.random_words failed, falling back to timestamp: {e}")
            return _timestamp_filename()
    logger.info("wonderwords unavailable, falling back to timestamp filename")
    return _timestamp_filename()


def _timestamp_filename() -> str:
    """Fallback filename: millisecond timestamp + 4-hex-char random suffix."""
    ts_ms = int(time.time() * 1000)
    suffix = os.urandom(2).hex()
    return f"generated-{ts_ms}-{suffix}.png"


try:
    from wonderwords import RandomWord as _RandomWord
    _random_word = _RandomWord()
except ImportError:
    _random_word = None


def generate_image(
    prompt: str,
    preset: dict,
    seed: int | None = None,
    output: str = "output.png",
    width: int = 512,
    height: int = 512,
    style: str = "ghibli",
    custom_styles: dict[str, str] | None = None,
    init_image: str | None = None,
    strength: float = 0.4,
    steps: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    model_name: str = DEFAULT_MODEL,
    model: Any = None,
) -> None:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Apply styling
    sm = StyleManager(custom_styles)
    styled_prompt = sm.apply_style(prompt, style)

    # Validate image-to-image parameters
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be between 0.0 and 1.0, got {strength}")

    if init_image is not None:
        from fluxgen.image_validation import validate_image_file
        # `validate_image_file` expands `~` and resolves symlinks/relative
        # components. Capture the resolved Path so the model receives
        # the canonical path rather than the raw CLI string.
        init_image = validate_image_file(init_image, label="reference image")

    spec = require_capability(model_name, "generate")
    if negative_prompt is not None and not spec.supports_negative_prompt:
        raise ValueError(
            f"Model '{spec.name}' does not support negative_prompt. "
            f"Models with negative-prompt support: "
            f"{', '.join(n for n, s in MODELS.items() if s.supports_negative_prompt)}"
        )
    # Explicit steps/guidance (MCP callers) take priority over the preset,
    # whose values in turn take priority over the spec defaults.
    steps, guidance = resolve_inference_params(
        spec, steps=steps, guidance=guidance, preset=preset
    )

    logger.info(f"Using model '{spec.name}' with {steps} steps, seed={seed}")

    # Use pre-loaded model or cache lookup
    if model is None:
        model = ModelManager.get_model(
            model_name=spec.name,
            quantize=preset.get("quantize"),
        )

    # Build generate_image kwargs — common across all models
    gen_kwargs = dict(
        seed=seed,
        prompt=styled_prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        image_path=init_image,
        image_strength=strength,
    )

    # Add guidance only for models that support it
    if guidance is not None:
        gen_kwargs["guidance"] = guidance

    # Negative conditioning — forwarded only when requested; ``None``
    # means "not requested" so the kwarg never reaches models whose
    # signature lacks it.
    if negative_prompt is not None:
        gen_kwargs["negative_prompt"] = negative_prompt

    # Generate the image
    result = model.generate_image(**gen_kwargs)

    image = result.image if hasattr(result, "image") else result

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")
