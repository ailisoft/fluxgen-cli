import logging
import random
from pathlib import Path

from fluxgen.models import (
    DEFAULT_EDIT_MODEL,
    ModelManager,
    require_capability,
    resolve_inference_params,
)

logger = logging.getLogger("fluxgen")

# Maximum dimension (longest side) for edit outputs. Larger inputs are
# downscaled to fit while preserving aspect ratio.
#
# This is the package-level default. Callers (CLI, MCP) may override
# per-call via ``ImageEditor.edit(..., max_dimension=...)`` or by reading
# ``max_edit_dimension`` from `.fluxgen.toml`. The CLI resolver supplies
# ``max_dimension=getattr(args, "max_edit_dimension", MAX_EDIT_DIMENSION)``.
MAX_EDIT_DIMENSION = 1920


class ImageEditor:
    """Instruction-based image editing via mflux Flux2-Klein-Edit."""

    def __init__(
        self,
        model_name: str = DEFAULT_EDIT_MODEL,
        quantize: int | None = None,
    ):
        self.spec = require_capability(model_name, "edit")
        self.model_name = self.spec.name
        self.quantize = quantize
        self.mflux_model = None

    def load(self) -> None:
        """Load the edit model into memory (idempotent)."""
        if self.mflux_model is not None:
            return
        quantize = self.quantize if self.quantize is not None else self.spec.default_quantize
        logger.info(f"Loading mflux model '{self.model_name}' on MLX...")
        self.mflux_model = ModelManager.get_model(
            model_name=self.model_name,
            quantize=quantize,
        )
        logger.debug("MFLUX model loaded successfully.")

    def _resolve_and_validate_inputs(
        self, image_paths: list[str]
    ) -> tuple[list[Path], tuple[int, int]]:
        """Resolve input paths, verify file integrity, and read the first image's size.

        Delegates each path's existence + ``is_file()`` + PIL integrity
        check to ``fluxgen.image_validation.validate_image_file`` (with
        ``label="input image"`` so error messages keep the pre-refactor
        "Input image not found" wording).

        Precondition: ``image_paths`` must be non-empty. argparse's
        ``nargs="+"`` guarantees this for the CLI path; calling with
        ``[]`` raises ``ValueError`` explicitly.

        Returns:
            ``(resolved_paths, first_image_size)`` where
            ``first_image_size`` is the ``(width, height)`` of
            ``image_paths[0]``.

        Raises:
            ValueError: ``image_paths`` is empty, or a path points at a
                directory rather than a file.
            FileNotFoundError: a path does not exist.
            InvalidImageError: a file is unreadable or fails integrity
                check.
        """
        from fluxgen.image_validation import validate_image_file

        if not image_paths:
            raise ValueError("image_paths must be non-empty")

        resolved_paths: list[Path] = []
        first_size: tuple[int, int] | None = None

        for idx, path in enumerate(image_paths):
            read_size = idx == 0
            if read_size:
                resolved, size = validate_image_file(
                    path, read_size=True, label="input image"
                )
                first_size = size
            else:
                resolved = validate_image_file(path, label="input image")
            resolved_paths.append(resolved)

        # Guaranteed by the empty-list guard above + the `if idx == 0` branch.
        assert first_size is not None
        return resolved_paths, first_size

    def edit(
        self,
        image_paths: list[str],
        prompt: str,
        output_path: str,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_dimension: int = MAX_EDIT_DIMENSION,
    ) -> None:
        """Perform instruction-based image editing.

        ``max_dimension`` caps the longest side of the edit output.
        Inputs or requested dimensions larger than this are downscaled
        while preserving aspect ratio. Defaults to
        :data:`MAX_EDIT_DIMENSION` (1920 px); the CLI passes the value
        from `[defaults] max_edit_dimension` in `.fluxgen.toml` when
        present, falling back to the default.
        """
        resolved_paths, (img_w, img_h) = self._resolve_and_validate_inputs(image_paths)

        # Log warning if multiple images are provided and we are using default size from first image
        if len(resolved_paths) > 1 and (width is None or height is None):
            logger.warning(
                f"Multiple input images provided. Defaulting dimensions to the first image's size: {img_w}x{img_h}."
            )

        run_width = width if width is not None else img_w
        run_height = height if height is not None else img_h

        # Limit maximum dimensions to the configured cap (per-call or
        # package default) while preserving aspect ratio.
        if run_width > max_dimension or run_height > max_dimension:
            aspect_ratio = run_width / run_height
            if run_width > run_height:
                new_w = max_dimension
                new_h = int(round(max_dimension / aspect_ratio))
            else:
                new_h = max_dimension
                new_w = int(round(max_dimension * aspect_ratio))

            if width is not None or height is not None:
                logger.warning(
                    f"Requested dimensions ({run_width}x{run_height}) exceed the {max_dimension}px limit. "
                    f"Downscaling overrides to {new_w}x{new_h} to preserve aspect ratio."
                )
            else:
                logger.warning(
                    f"Input image dimensions ({img_w}x{img_h}) exceed the {max_dimension}px limit. "
                    f"Downscaling to {new_w}x{new_h} to preserve aspect ratio."
                )
            run_width, run_height = new_w, new_h

        self.load()

        run_steps, run_guidance = resolve_inference_params(
            self.spec,
            steps=steps,
            guidance=guidance_scale,
        )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        logger.debug(
            f"Editing images with prompt: '{prompt}' using {self.model_name} "
            f"(seed={seed}, steps={run_steps}, guidance={run_guidance}, "
            f"size={run_width}x{run_height})..."
        )
        result_wrapper = self.mflux_model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=run_steps,
            guidance=run_guidance,
            image_paths=resolved_paths,
            width=run_width,
            height=run_height,
        )

        if hasattr(result_wrapper, "image"):
            result = result_wrapper.image
        else:
            result = result_wrapper

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.info(f"Edited image saved to {out_path}")
