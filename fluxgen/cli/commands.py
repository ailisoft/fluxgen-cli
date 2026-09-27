"""Command handlers and their subparser definitions.

This module owns everything bound to a subcommand: the argparse
subparser definitions (via ``add_generate_parser`` and
``add_edit_parser``), the handlers that run them (``handle_generate``
and ``handle_edit``), the output-path resolver, the error-handling
context manager, and the resolution priority chain
(:func:`resolve_image_dimensions`).

Why split this off: the generate and edit flows are the only place
that imports the heavyweight ``fluxgen.editor`` and
``fluxgen.generator`` modules (model loaders, mflux). By isolating
that here, ``cli/__init__.py`` stays light enough to import during
``--help`` / ``--version`` fast-paths without paying the
heavy-import cost, and ``handle_interactive`` can import only what
it actually needs.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from fluxgen.config import get_config_value
from fluxgen.editor import MAX_EDIT_DIMENSION
from fluxgen.exceptions import FluxgenError, PathTraversalError
from fluxgen.generator import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    ModelManager,
    generate_image,
    generate_random_filename,
)
from fluxgen.models import (
    DEFAULT_EDIT_MODEL,
    SUPPORTED_EDIT_MODELS,
    get_model_spec,
    resolve_quantize,
)
from fluxgen.presets import ALL_RESOLUTION_PRESETS, PRESETS, PRESETS_BY_NAME

from fluxgen.cli.presets_arg import (
    add_preset_args,
    add_resolution_args,
)

logger = logging.getLogger("fluxgen")


# ── Resolution priority chain ──────────────────────────────────────────────
#
# Public so it can be tested in isolation. The full priority order
# (highest to lowest) is documented in the function's docstring; the
# short version is: per-axis CLI flag > explicit --resolution >
# config > 512 default. When only one axis is passed, the missing
# axis falls through the same chain independently.

_DEFAULT_DIMENSION = 512


def resolve_image_dimensions(args, config) -> tuple[int, int]:
    """Compute ``(width, height)`` from CLI args + config.

    Priority (highest to lowest):

    1. **Explicit ``--width`` / ``--height`` flags** — each axis is
       resolved independently. If only one axis is passed, the other
       falls through to ``--resolution`` (when set) > config >
       :data:`_DEFAULT_DIMENSION`.
    2. **``--resolution`` preset** — applied to both axes when no
       explicit width/height was passed. Always overrides config.
    3. **Config ``width`` / ``height``** — from
       ``.fluxgen.toml`` ``[defaults]``.
    4. **Hard default** — :data:`_DEFAULT_DIMENSION` (``512``) per
       axis.

    The function reads attribute presence via ``getattr(..., None)``
    rather than truthy checks because ``0`` is a legal (if silly)
    width. argparse uses ``argparse.SUPPRESS`` as the default for
    these flags, which omits the attribute entirely from the parsed
    namespace; ``getattr(..., None)`` then resolves to ``None`` for
    "user passed nothing" while preserving an explicit ``0`` as
    ``0``.
    """
    cli_resolution = getattr(args, "resolution", None)
    cli_width = getattr(args, "width", None)
    cli_height = getattr(args, "height", None)
    config_width = get_config_value(config, "width", None)
    config_height = get_config_value(config, "height", None)

    # Unpack the preset once so both axes can pull their respective
    # element without a second dict lookup. ``None`` when no preset
    # was passed, in which case _resolve_axis skips the preset layer.
    preset = ALL_RESOLUTION_PRESETS.get(cli_resolution) if cli_resolution else None
    preset_width = preset[0] if preset else None
    preset_height = preset[1] if preset else None

    if cli_width is not None or cli_height is not None:
        # Explicit per-axis CLI flag wins. Missing axis falls through
        # the rest of the chain via _resolve_axis.
        width = _resolve_axis(cli_width, preset_width, config_width)
        height = _resolve_axis(cli_height, preset_height, config_height)
    elif preset is not None:
        # Explicit --resolution overrides config (no CLI w/h set).
        width, height = preset
    else:
        # No CLI flags: fall back to config, then default.
        width = config_width if config_width is not None else _DEFAULT_DIMENSION
        height = config_height if config_height is not None else _DEFAULT_DIMENSION

    return width, height


def _resolve_axis(cli_value, preset_value, config_value):
    """Resolve a single axis (width or height) through the priority chain.

    Priority (highest to lowest):

    1. Explicit CLI value (``--width`` / ``--height``).
    2. Resolution preset element for this axis (``preset_value`` is
       already pre-unpacked by the caller, so this helper has no
       knowledge of which axis it is — it just falls through the
       chain).
    3. Config value from ``.fluxgen.toml``.
    4. Hard default.

    Shared by both axes inside :func:`resolve_image_dimensions` when
    an explicit per-axis CLI flag was passed for at least one axis;
    the missing axis falls through the same chain independently.
    """
    if cli_value is not None:
        return cli_value
    if preset_value is not None:
        return preset_value
    if config_value is not None:
        return config_value
    return _DEFAULT_DIMENSION


# ── Subparser builders ─────────────────────────────────────────────────────


def add_generate_parser(subparsers, verbosity_parent, config):
    """Attach the ``generate`` / ``gen`` subparser to ``subparsers``.

    ``verbosity_parent`` is the shared parent that carries
    ``-v/--verbose`` and ``-s/--silent`` so help text doesn't drift
    across subcommands; ``config`` supplies defaults for
    ``--output-dir``, ``--style``, and ``--model``.
    """
    gen_parser = subparsers.add_parser(
        "generate",
        aliases=["gen"],
        help="Generate an image from text",
        parents=[verbosity_parent],
    )
    gen_parser.add_argument("prompt", help="Text prompt for image generation")

    add_preset_args(gen_parser)

    gen_parser.add_argument("--steps", type=int, help="Override steps")
    gen_parser.add_argument("--quantize", type=int, help="Override quantize")
    gen_parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
    gen_parser.add_argument(
        "--output-dir", type=str,
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory (default: output)",
    )
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument(
        "--style", type=str,
        default=get_config_value(config, "style", "none"),
        help="Style to apply (default: none)",
    )
    gen_parser.add_argument(
        "--no-style", action="store_const", const="none", dest="style",
        help="Disable styling",
    )
    gen_parser.add_argument(
        "--model", type=str, choices=SUPPORTED_MODELS,
        default=get_config_value(config, "model", DEFAULT_MODEL),
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )

    add_resolution_args(gen_parser)

    gen_parser.add_argument("--init-image", type=str, help="Reference image for img2img")
    gen_parser.add_argument("--strength", type=float, default=0.4, help="Img2img strength")
    gen_parser.add_argument(
        "--no-timer", action="store_false", dest="timer", default=True,
        help="Hide generation time",
    )
    return gen_parser


def add_edit_parser(subparsers, verbosity_parent, config):
    """Attach the ``edit`` subparser to ``subparsers``."""
    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit an image using natural-language instructions",
        parents=[verbosity_parent],
    )
    edit_parser.add_argument("image", nargs="+", help="Path to the input image(s)")
    edit_parser.add_argument(
        "prompt", help="Instruction for the edit (e.g., 'add a red hat')"
    )
    edit_parser.add_argument("--output", help="Output filename (saved in output dir)")
    edit_parser.add_argument(
        "--output-dir", type=str,
        default=get_config_value(config, "output_dir", "output"),
        help="Output directory (default: output)",
    )
    edit_parser.add_argument(
        "--model", type=str, choices=list(SUPPORTED_EDIT_MODELS),
        default=DEFAULT_EDIT_MODEL,
        help=f"Model to use for editing (default: {DEFAULT_EDIT_MODEL})",
    )
    edit_parser.add_argument(
        "--quantize", type=int,
        help=f"Override quantize for {DEFAULT_EDIT_MODEL}",
    )
    edit_parser.add_argument("--seed", type=int, help="Random seed")
    edit_parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    edit_parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    edit_parser.add_argument(
        "--width", type=int,
        help="Output image width (defaults to input image width)",
    )
    edit_parser.add_argument(
        "--height", "--length", type=int, dest="height",
        help="Output image height/length (defaults to input image height/length)",
    )
    edit_parser.add_argument(
        "--no-timer", action="store_false", dest="timer", default=True,
        help="Hide execution time",
    )
    return edit_parser


# ── Output path resolver ───────────────────────────────────────────────────


def resolve_output_path(output_arg, output_dir_arg, default_filename_func=None):
    """Resolve the absolute output path for a generation or edit.

    Behavior:

    - If ``output_arg`` is an absolute path, use it verbatim.
    - If ``output_arg`` is a relative path, resolve it under
      ``output_dir_arg``. The result must remain inside
      ``output_dir_arg`` after resolution; anything outside raises
      :class:`PathTraversalError`. This protects against
      ``--output ../../etc/passwd`` style escapes even when the user
      passes a non-traversally-suspicious-looking relative path.
    - If ``output_arg`` is empty, generate a filename via
      ``default_filename_func`` (or
      :func:`fluxgen.generator.generate_random_filename` when no
      function is provided) and resolve it under the output dir.

    Returns the resolved path as a string (so callers can hand it
    directly to PIL / pipeline save APIs without re-stringifying).
    """
    output_dir = Path(output_dir_arg).expanduser().resolve()

    if output_arg:
        output_path = Path(output_arg)
        if output_path.is_absolute():
            final_path = output_path.resolve()
        else:
            final_path = (output_dir / output_path).resolve()
            if not final_path.is_relative_to(output_dir):
                raise PathTraversalError(
                    f"Output path {final_path} is outside the allowed directory {output_dir}"
                )
    else:
        if default_filename_func:
            output_filename = default_filename_func()
        else:
            output_filename = generate_random_filename()
        final_path = (output_dir / output_filename).resolve()

    return str(final_path)


# ── Error handling context manager ─────────────────────────────────────────


@contextmanager
def error_handler(args, interactive=False):
    """Convert expected exceptions into ``logger.error`` + optional exit.

    Catches :class:`FileNotFoundError`, :class:`ValueError`, and
    :class:`fluxgen.exceptions.FluxgenError` as "expected" failures
    (single-line error, no traceback unless ``--verbose``), and
    treats any other :class:`Exception` as unexpected (still logged
    with a single-line message, but traceback is suppressed unless
    verbose). Outside the REPL, exits with status 1 after logging;
    in the REPL, the loop catches the raised exception and continues.
    """
    try:
        yield
    except (FileNotFoundError, ValueError, FluxgenError) as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if not interactive:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if not interactive:
            sys.exit(1)


# ── Handlers ───────────────────────────────────────────────────────────────


def handle_generate(args, config, interactive=False):
    """Run the generation pipeline for parsed ``args``.

    Steps: resolve preset index (CLI ``--preset`` wins over the
    numeric shortcuts; falls back to config when neither is set),
    apply per-flag overrides (``--steps``, ``--quantize``), build
    the output path, compute output dimensions via the priority
    chain in :func:`resolve_image_dimensions`, pre-load the model
    so the timer reflects only inference (not weight loading), and
    finally call :func:`fluxgen.generator.generate_image`.
    """
    with error_handler(args, interactive):
        # Determine preset index
        preset_idx = args.preset_idx
        if args.preset:
            preset_idx = PRESETS_BY_NAME[args.preset]

        if preset_idx is None:
            preset_idx = get_config_value(config, "preset", 0)

        preset = asdict(PRESETS[preset_idx])
        if args.steps:
            preset["steps"] = args.steps
        # Quantization: explicit --quantize → per-model default → preset.
        # The resolved value is written back into the preset so the
        # preloaded model and generate_image agree on the cache key.
        preset["quantize"] = resolve_quantize(
            get_model_spec(args.model), preset, args.quantize
        )

        output_path = resolve_output_path(args.output, args.output_dir)
        width, height = resolve_image_dimensions(args, config)

        # Pre-load model before timer starts
        preloaded_model = ModelManager.get_model(
            model_name=args.model,
            quantize=preset.get("quantize"),
        )

        start = time.perf_counter() if getattr(args, "timer", True) else None
        generate_image(
            prompt=args.prompt,
            preset=preset,
            seed=args.seed,
            output=output_path,
            width=width,
            height=height,
            style=args.style,
            custom_styles=config.get("styles"),
            init_image=args.init_image,
            strength=args.strength,
            model_name=args.model,
            model=preloaded_model,
        )
        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(f"\u23a1 Generated in {elapsed:.2f}s")


def handle_edit(args, config=None, interactive=False):
    """Run the edit pipeline for parsed ``args``.

    Resolves the output ``max_dimension`` cap from the config (with
    type-check + warning on invalid values, falling back to the
    package default), runs cheap pre-flight existence checks on
    every input image so we fail-fast before loading the model,
    and finally constructs an :class:`fluxgen.editor.ImageEditor`
    and calls :meth:`ImageEditor.edit`. ``config`` is optional so
    test code can call ``handle_edit(args)`` without constructing a
    full config dict.
    """
    with error_handler(args, interactive):
        # ``ImageEditor`` is imported lazily here rather than at the
        # top of the module. ``fluxgen.editor`` is already loaded by
        # the time we reach this function (MAX_EDIT_DIMENSION is a
        # top-level import), so the lazy import isn't about deferred
        # loading — it's about test isolation: tests patch
        # ``fluxgen.editor.ImageEditor`` via ``with patch(...)`` and
        # the lazy import resolves through the module attribute at
        # call time, picking up the mock. A top-level
        # ``from fluxgen.editor import ImageEditor`` would capture
        # the real class at module load time and the patch wouldn't
        # take effect.
        from fluxgen.editor import ImageEditor

        # Resolve max_dimension from config, falling back to the
        # package default. ``config`` is optional so the test suite
        # can call ``handle_edit(args)`` without constructing a full
        # config dict; when omitted, the package default applies.
        max_dimension = get_config_value(
            config or {}, "max_edit_dimension", MAX_EDIT_DIMENSION
        )
        # Type-check the config value — a malformed entry (e.g. a
        # string) should not silently bypass the cap. Use ``type() is
        # int`` rather than ``isinstance(..., int)`` so a TOML
        # ``max_edit_dimension = true`` doesn't sneak through:
        # ``bool`` is a subclass of ``int``, so ``isinstance(True,
        # int)`` is True, but the editor expects a real int (it
        # compares dimensions with ``>`` and arithmetic). Rejecting
        # bool keeps the contract honest.
        if type(max_dimension) is not int or max_dimension <= 0:
            logger.warning(
                f"Ignoring invalid 'max_edit_dimension'={max_dimension!r} in config; "
                f"using default {MAX_EDIT_DIMENSION}."
            )
            max_dimension = MAX_EDIT_DIMENSION

        # Cheap path checks (existence + is_file) duplicated from
        # ``ImageEditor._resolve_and_validate_inputs``: lets us
        # fail-fast before constructing the editor + loading the
        # model below. The editor re-runs these plus a full PIL
        # integrity check.
        input_paths = [Path(img).expanduser().resolve() for img in args.image]
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            if not p.is_file():
                raise ValueError(f"Input image must be a file: {p}")

        def generate_edit_filename():
            base_name = input_paths[0].stem
            try:
                from wonderwords import RandomWord
                rw = RandomWord()
                random_word = rw.random_words(1, word_max_length=5)[0]
                return f"{base_name}_{random_word}.png"
            except ImportError:
                # ``generate_random_filename`` is imported at the top
                # of this module — use that reference in the
                # wonderwords-missing fallback rather than re-importing.
                return generate_random_filename()

        output_path = resolve_output_path(args.output, args.output_dir, generate_edit_filename)

        model_name = getattr(args, "model", DEFAULT_EDIT_MODEL)
        quantize = getattr(args, "quantize", None)
        seed = getattr(args, "seed", None)

        editor = ImageEditor(model_name=model_name, quantize=quantize)

        # Pre-load model before timer starts
        editor.load()

        logger.info(f"Applying edit: '{args.prompt}'")
        start = time.perf_counter() if getattr(args, "timer", True) else None
        editor.edit(
            image_paths=args.image,
            prompt=args.prompt,
            output_path=output_path,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed,
            width=args.width,
            height=args.height,
            max_dimension=max_dimension,
        )

        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(f"\u23a1 Edited in {elapsed:.2f}s")
