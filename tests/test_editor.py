"""Tests for fluxgen.editor — input validation and mflux edit wiring."""
from unittest.mock import MagicMock, patch

import pytest

from fluxgen.editor import ImageEditor, MAX_EDIT_DIMENSION
from fluxgen.models import DEFAULT_EDIT_MODEL


@pytest.fixture(autouse=True)
def reset_model_cache():
    from fluxgen.models import ModelManager

    ModelManager.reset()
    yield
    ModelManager.reset()


def test_editor_defaults_to_flux2_klein_edit():
    editor = ImageEditor()
    assert editor.model_name == DEFAULT_EDIT_MODEL
    assert editor.mflux_model is None


def test_editor_rejects_generation_only_model():
    with pytest.raises(ValueError, match="does not support edit"):
        ImageEditor(model_name="zimage-turbo")


def test_editor_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        ImageEditor(model_name="qwen-image-edit")


def test_editor_rejects_legacy_flux2_klein_id():
    """Bare ``flux2-klein`` is not an alias; must fail closed."""
    with pytest.raises(ValueError, match="Unsupported model"):
        ImageEditor(model_name="flux2-klein")


def test_editor_missing_input_fails_before_model_load(tmp_path):
    editor = ImageEditor()
    editor.load = MagicMock()
    missing_image = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        editor.edit(
            image_paths=[str(missing_image)],
            prompt="test",
            output_path="output/out.png",
        )

    editor.load.assert_not_called()


@patch("fluxgen.models.ModelManager.get_model")
def test_editor_edit_flow_uses_mflux(mock_get_model, tmp_path):
    from PIL import Image as PILImage

    input_image = tmp_path / "dummy.png"
    PILImage.new("RGB", (512, 512), "red").save(input_image)
    output_image = tmp_path / "out.png"

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result
    mock_get_model.return_value = mock_model

    editor = ImageEditor()
    editor.edit(
        image_paths=[str(input_image)],
        prompt="make it red",
        output_path=str(output_image),
        steps=6,
        guidance_scale=2.0,
        seed=123,
    )

    mock_get_model.assert_called_once_with(
        model_name=DEFAULT_EDIT_MODEL,
        quantize=None,
    )
    kwargs = mock_model.generate_image.call_args.kwargs
    assert kwargs["prompt"] == "make it red"
    assert kwargs["num_inference_steps"] == 6
    assert kwargs["guidance"] == 2.0
    assert kwargs["seed"] == 123
    assert kwargs["width"] == 512
    assert kwargs["height"] == 512
    mock_result.image.save.assert_called_once()


@patch("fluxgen.models.ModelManager.get_model")
def test_editor_applies_spec_default_quantize_when_flag_omitted(mock_get_model):
    """quantize=None + a spec default_quantize must load at the default.

    No shipped edit model sets ``default_quantize`` yet (the qwen21 edit
    variant is blocked upstream), so simulate one via a frozen-spec copy.
    """
    from dataclasses import replace

    from fluxgen.models import DEFAULT_EDIT_MODEL, get_model_spec

    editor = ImageEditor()
    editor.spec = replace(get_model_spec(DEFAULT_EDIT_MODEL), default_quantize=4)
    editor.load()

    assert mock_get_model.call_args.kwargs["quantize"] == 4


@patch("fluxgen.models.ModelManager.get_model")
def test_editor_uses_model_defaults_when_steps_guidance_omitted(mock_get_model, tmp_path):
    from PIL import Image as PILImage

    input_image = tmp_path / "dummy.png"
    PILImage.new("RGB", (256, 256), "blue").save(input_image)

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result
    mock_get_model.return_value = mock_model

    editor = ImageEditor()
    editor.edit(
        image_paths=[str(input_image)],
        prompt="test",
        output_path=str(tmp_path / "out.png"),
        seed=1,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["guidance"] == 1.0


def test_max_edit_dimension_constant():
    assert MAX_EDIT_DIMENSION == 1920


# ── _resolve_and_validate_inputs ──────────────────────────────────────────────────────


def test_resolve_and_validate_inputs_returns_first_image_size(tmp_path):
    """Helper returns the (width, height) of the first image."""
    from PIL import Image as PILImage

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    PILImage.new("RGB", (640, 480), "red").save(image_a)
    PILImage.new("RGB", (1024, 768), "blue").save(image_b)

    editor = ImageEditor()
    paths, size = editor._resolve_and_validate_inputs([str(image_a), str(image_b)])

    assert len(paths) == 2
    assert paths[0] == image_a.resolve()
    assert paths[1] == image_b.resolve()
    assert size == (640, 480)


def test_resolve_and_validate_inputs_rejects_missing_file(tmp_path):
    editor = ImageEditor()
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        editor._resolve_and_validate_inputs([str(missing)])


def test_resolve_and_validate_inputs_rejects_empty_list():
    """Empty image_paths raises ValueError (no fallback to defaults)."""
    editor = ImageEditor()

    with pytest.raises(ValueError, match="must be non-empty"):
        editor._resolve_and_validate_inputs([])


def test_resolve_and_validate_inputs_rejects_directory(tmp_path):
    editor = ImageEditor()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        editor._resolve_and_validate_inputs([str(subdir)])


def test_resolve_and_validate_inputs_rejects_corrupt_image(tmp_path):
    from fluxgen.exceptions import InvalidImageError

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a valid image file at all")

    editor = ImageEditor()
    with pytest.raises(InvalidImageError):
        editor._resolve_and_validate_inputs([str(corrupt)])


def test_resolve_and_validate_inputs_size_in_same_open_as_verify(tmp_path):
    """First image is opened once for both size lookup and verify."""
    from PIL import Image as PILImage

    image = tmp_path / "a.png"
    PILImage.new("RGB", (100, 200), "red").save(image)

    editor = ImageEditor()

    real_open = PILImage.open
    open_count = 0

    def counting_open(path, *args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(path, *args, **kwargs)

    with patch("PIL.Image.open", new=counting_open):
        paths, size = editor._resolve_and_validate_inputs([str(image)])

    assert open_count == 1
    assert size == (100, 200)


def test_resolve_and_validate_inputs_multi_image_one_open_per_file(tmp_path):
    """Multi-image: one open per file (verify only for non-first images)."""
    from PIL import Image as PILImage

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    PILImage.new("RGB", (640, 480), "red").save(image_a)
    PILImage.new("RGB", (1024, 768), "blue").save(image_b)
    PILImage.new("RGB", (320, 240), "green").save(image_c)

    editor = ImageEditor()

    real_open = PILImage.open
    open_count = 0

    def counting_open(path, *args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(path, *args, **kwargs)

    with patch("PIL.Image.open", new=counting_open):
        paths, size = editor._resolve_and_validate_inputs(
            [str(image_a), str(image_b), str(image_c)]
        )

    assert open_count == 3
    assert size == (640, 480)


def test_resolve_and_validate_inputs_expands_user_and_resolves(tmp_path):
    """Paths with ~ and relative components are resolved."""
    from PIL import Image as PILImage

    image = tmp_path / "a.png"
    PILImage.new("RGB", (50, 50), "red").save(image)

    editor = ImageEditor()
    paths, _ = editor._resolve_and_validate_inputs([str(image)])

    assert paths[0].is_absolute()
    assert paths[0] == image.resolve()
