"""Tests for fluxgen.generator helpers — filename generation, model registry."""
import re
from unittest.mock import MagicMock, patch

import pytest

from fluxgen.generator import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    _timestamp_filename,
    generate_random_filename,
)
from fluxgen.models import (
    DEFAULT_EDIT_MODEL,
    MODELS,
    SUPPORTED_EDIT_MODELS,
    SUPPORTED_GENERATION_MODELS,
    get_model_spec,
    require_capability,
    resolve_inference_params,
)


# ── generate_random_filename ────────────────────────────────────────────────────


def test_generate_random_filename_uses_wonderwords_when_available():
    """Happy path: wonderwords returns 3 hyphenated words + .png."""
    fn = generate_random_filename()
    # 3 short words joined by hyphens
    assert re.match(r"^[a-z]+(-[a-z]+){2}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_fallback_when_wonderwords_unavailable():
    """When wonderwords isn't installed, fall back to timestamp+suffix."""
    with patch("fluxgen.generator._random_word", None):
        fn = generate_random_filename()

    # ms timestamp (13 digits) + 4 hex chars + .png
    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_fallback_when_random_words_raises():
    """If wonderwords.random_words raises, fall back to timestamp+suffix."""
    fake_rw = patch("fluxgen.generator._random_word").start()
    fake_rw.random_words.side_effect = RuntimeError("boom")
    try:
        fn = generate_random_filename()
    finally:
        patch.stopall()

    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_generate_random_filename_no_collision_under_load():
    """100 consecutive fallback calls produce 100 unique filenames."""
    with patch("fluxgen.generator._random_word", None):
        fns = [generate_random_filename() for _ in range(100)]

    assert len(set(fns)) == 100, f"only {len(set(fns))}/100 unique filenames"


def test_generate_random_filename_unique_for_same_millisecond():
    """Two calls in the same millisecond must still differ (via random suffix)."""
    with patch("fluxgen.generator._random_word", None), \
         patch("fluxgen.generator.time.time", return_value=1_700_000_000.123):
        fns = {generate_random_filename() for _ in range(20)}

    assert len(fns) == 20, "Same-millisecond calls collided"


# ── _timestamp_filename ────────────────────────────────────────────────────────


def test_timestamp_filename_format():
    """Format: generated-{ms}-{4hex}.png"""
    fn = _timestamp_filename()
    assert re.match(r"^generated-\d{13}-[0-9a-f]{4}\.png$", fn), f"unexpected: {fn}"


def test_timestamp_filename_uses_millisecond_timestamp():
    """Timestamp should be milliseconds (13 digits) not seconds (10)."""
    with patch("fluxgen.generator.time.time", return_value=1_700_000_000.123):
        fn = _timestamp_filename()

    # 1_700_000_000.123 * 1000 = 1_700_000_000_123 (13 digits)
    assert "1700000000123" in fn


def test_timestamp_filename_unique_across_calls():
    """Direct test of the helper: 50 calls produce 50 unique names."""
    fns = {_timestamp_filename() for _ in range(50)}
    assert len(fns) == 50


# ── Model registry ─────────────────────────────────────────────────────────────


def test_supported_models_are_non_empty_strings():
    assert len(SUPPORTED_MODELS) >= 1
    for m in SUPPORTED_MODELS:
        assert isinstance(m, str)
        assert m  # non-empty


def test_default_model_is_in_supported_models():
    assert DEFAULT_MODEL in SUPPORTED_MODELS


def test_generation_and_edit_lists_partition_registry():
    assert set(SUPPORTED_GENERATION_MODELS) | set(SUPPORTED_EDIT_MODELS) == set(MODELS)
    assert set(SUPPORTED_GENERATION_MODELS).isdisjoint(SUPPORTED_EDIT_MODELS)
    assert DEFAULT_EDIT_MODEL in SUPPORTED_EDIT_MODELS
    assert DEFAULT_EDIT_MODEL not in SUPPORTED_GENERATION_MODELS


def test_resolve_inference_params_uses_spec_when_preset_guidance_is_none():
    """Preset dataclasses always serialize guidance=None; that must not
    shadow the model default.
    """
    spec = get_model_spec("zimage")
    steps, guidance = resolve_inference_params(
        spec,
        preset={"steps": 9, "guidance": None, "quantize": 8},
    )
    assert steps == 9
    assert guidance == 4.0


def test_resolve_inference_params_skips_guidance_for_turbo():
    spec = get_model_spec("zimage-turbo")
    steps, guidance = resolve_inference_params(
        spec,
        preset={"steps": None, "guidance": None},
    )
    assert steps == 4
    assert guidance is None


def test_resolve_inference_params_explicit_kwargs_win():
    spec = get_model_spec("flux2-klein9b")
    steps, guidance = resolve_inference_params(
        spec,
        steps=12,
        guidance=2.5,
        preset={"steps": 4, "guidance": 3.5},
    )
    assert steps == 12
    assert guidance == 2.5


def test_generate_image_passes_model_default_guidance_when_preset_none(tmp_path):
    """End-to-end: ``asdict(Preset)`` guidance=None must become zimage's 4.0."""
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    out = tmp_path / "out.png"
    generate_image(
        prompt="a fox",
        preset={"steps": 9, "guidance": None, "quantize": 8},
        seed=1,
        output=str(out),
        width=64,
        height=64,
        style="none",
        model_name="zimage",
        model=mock_model,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert kwargs["guidance"] == 4.0
    assert kwargs["num_inference_steps"] == 9
    mock_result.image.save.assert_called_once()


def test_generate_image_omits_guidance_for_turbo(tmp_path):
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": 4, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="zimage-turbo",
        model=mock_model,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert "guidance" not in kwargs


def test_require_capability_rejects_wrong_capability():
    with pytest.raises(ValueError, match="does not support edit"):
        require_capability("zimage", "edit")
    with pytest.raises(ValueError, match="does not support generate"):
        require_capability(DEFAULT_EDIT_MODEL, "generate")


def test_krea2_is_generate_only_with_turbo_defaults():
    """Krea 2 Turbo is an 8-step-distilled txt2img model (CFG 1.0)."""
    spec = get_model_spec("krea2")
    assert spec.capabilities == {"generate"}
    assert spec.steps == 8
    assert spec.guidance == 1.0
    assert "krea2" not in SUPPORTED_EDIT_MODELS
    with pytest.raises(ValueError, match="does not support edit"):
        require_capability("krea2", "edit")


def test_resolve_inference_params_krea2_spec_defaults():
    spec = get_model_spec("krea2")
    steps, guidance = resolve_inference_params(
        spec,
        preset={"steps": None, "guidance": None, "quantize": 8},
    )
    assert steps == 8
    assert guidance == 1.0


def test_generate_image_passes_krea2_default_guidance(tmp_path):
    """End-to-end: a preset whose steps are None must not lose Krea 2's 8 steps
    or its 1.0 CFG default.
    """
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": None, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="krea2",
        model=mock_model,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert kwargs["guidance"] == 1.0
    assert kwargs["num_inference_steps"] == 8
    mock_result.image.save.assert_called_once()


def test_qwen21_is_generate_only_with_guidance_free_defaults():
    """Qwen-Image-2.1 is a 40-step guidance-free txt2img model on mflux >= 0.20.0."""
    spec = get_model_spec("qwen21")
    assert spec.capabilities == {"generate"}
    assert spec.steps == 40
    assert spec.guidance is None
    assert spec.default_quantize == 4
    assert "qwen21" not in SUPPORTED_EDIT_MODELS
    with pytest.raises(ValueError, match="does not support edit"):
        require_capability("qwen21", "edit")


def test_resolve_inference_params_qwen21_ignores_preset_guidance():
    """guidance=None specs must hard-disable guidance, even if a preset carries one."""
    spec = get_model_spec("qwen21")
    steps, guidance = resolve_inference_params(
        spec,
        preset={"steps": None, "guidance": 3.5, "quantize": 8},
    )
    assert steps == 40
    assert guidance is None


def test_qwen21_viggle_turbo_registered_as_generation_model():
    """Viggle's 4-step turbo LoRA: generation-only, CFG-free, q4 by default."""
    spec = get_model_spec("qwen21-viggle-turbo")
    assert spec.capabilities == {"generate"}
    assert spec.steps == 4
    assert spec.guidance is None
    assert spec.min_steps == 4
    assert spec.default_quantize == 4
    assert "qwen21-viggle-turbo" in SUPPORTED_GENERATION_MODELS
    assert "qwen21-viggle-turbo" not in SUPPORTED_EDIT_MODELS
    with pytest.raises(ValueError, match="does not support edit"):
        require_capability("qwen21-viggle-turbo", "edit")


def test_resolve_inference_params_short_circuits_below_min_steps():
    """Step-distilled specs reject sub-floor step counts from kwargs or presets."""
    spec = get_model_spec("qwen21-viggle-turbo")
    for low in (1, 2, 3):
        with pytest.raises(ValueError, match="sweet spot"):
            resolve_inference_params(spec, steps=low)
        with pytest.raises(ValueError, match="sweet spot"):
            resolve_inference_params(spec, preset={"steps": low, "guidance": None})


def test_resolve_inference_params_allows_min_steps_and_above():
    spec = get_model_spec("qwen21-viggle-turbo")
    assert resolve_inference_params(spec, steps=4) == (4, None)
    assert resolve_inference_params(spec, steps=8) == (8, None)
    assert resolve_inference_params(spec, preset={"steps": None, "guidance": None}) == (4, None)


def test_min_steps_ignores_specs_without_a_floor():
    """Existing specs keep their behavior: no min_steps, no short-circuit."""
    spec = get_model_spec("zimage-turbo")
    assert spec.min_steps is None
    steps, guidance = resolve_inference_params(spec, steps=1)
    assert steps == 1
    assert guidance is None


def test_qwen21_lora_mapping_covers_viggle_adapter_keys():
    """Every key shape in the real Viggle r64 adapter header (454 tensors,
    'transformer.'-prefixed lora_A/lora_B) matches exactly one mapping
    target, and {block} patterns resolve to the right MLX module path.
    """
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader

    from fluxgen.qwen21_lora_mapping import Qwen21LoRAMapping

    mappings = LoRALoader._build_pattern_mappings(Qwen21LoRAMapping.get_mapping())

    def matched_targets(key: str) -> list[str]:
        hits = [
            m.target_path for m in mappings if LoRALoader._match_pattern(key, m.source_pattern) is not None
        ]
        return hits

    block_modules = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.proj",
        "img_mlp.out",
        "img_mlp.gate_layer",
    )
    for block in range(Qwen21LoRAMapping.NUM_TRANSFORMER_BLOCKS):
        for module in block_modules:
            mlx_path = f"transformer_blocks.{block}.{module}"
            for suffix in ("lora_A.weight", "lora_B.weight"):
                key = f"transformer.{mlx_path}.{suffix}"
                hits = [h.replace("{block}", str(block)) for h in matched_targets(key)]
                assert hits == [mlx_path], f"{key}: {hits}"

    # Checkpoint spelling → MLX module path. modulation is the known
    # divergence: checkpoint modulation.1, MLX module modulation.layers.1.
    global_modules = (
        "modulation.1",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
    )
    for module in global_modules:
        for suffix in ("lora_A.weight", "lora_B.weight"):
            key = f"transformer.{module}.{suffix}"
            hits = matched_targets(key)
            expected = "modulation.layers.1" if module == "modulation.1" else module
            assert hits == [expected], f"{key}: {hits}"


def test_qwen21_lora_mapping_targets_resolve_on_real_transformer():
    """Pattern matching alone can't catch an unresolvable model_path (the
    loader only fails at apply time), so assert every mapping target
    resolves to a linear module on a real Qwen21Transformer.
    """
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from fluxgen.qwen21_lora_mapping import Qwen21LoRAMapping

    transformer = Qwen21Transformer()
    for target in Qwen21LoRAMapping.get_mapping():
        module_path = target.model_path.replace("{block}", "0")
        module = LoRALoader._get_target_module(transformer, module_path)
        assert hasattr(module, "weight"), f"{target.model_path} is not a linear module"


def test_generate_image_omits_guidance_for_qwen21(tmp_path):
    """End-to-end: qwen21 never receives a guidance kwarg (mflux defaults CFG 1.0)."""
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": None, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="qwen21",
        model=mock_model,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert "guidance" not in kwargs
    assert kwargs["num_inference_steps"] == 40
    mock_result.image.save.assert_called_once()


# ── resolve_quantize ───────────────────────────────────────────────────────────


def test_resolve_quantize_priority_chain():
    """Explicit flag → model default → preset → bf16 (None)."""
    from fluxgen.models import resolve_quantize

    spec = get_model_spec("qwen21")  # default_quantize=4
    # Explicit flag wins over the model default.
    assert resolve_quantize(spec, {"quantize": 8}, cli_quantize=8) == 8
    assert resolve_quantize(spec, {"quantize": 8}, cli_quantize=16) == 16
    # Model default beats the preset value.
    assert resolve_quantize(spec, {"quantize": 8}) == 4
    # Models without a default fall through to the preset.
    zimage_spec = get_model_spec("zimage")
    assert resolve_quantize(zimage_spec, {"quantize": 8}) == 8
    assert resolve_quantize(zimage_spec, {"quantize": None}) is None
    assert resolve_quantize(zimage_spec, None) is None


def test_generate_image_applies_qwen21_default_quantize(tmp_path):
    """model=None path (MCP): the spec's q4 default must beat the preset's q8."""
    from fluxgen.generator import generate_image
    from fluxgen.models import ModelManager

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    with patch.object(
        ModelManager, "get_model", return_value=mock_model
    ) as mock_get:
        generate_image(
            prompt="a fox",
            preset={"steps": None, "guidance": None, "quantize": 8},
            seed=1,
            output=str(tmp_path / "out.png"),
            width=64,
            height=64,
            style="none",
            model_name="qwen21",
        )

    assert mock_get.call_args.kwargs["quantize"] == 4
    mock_model.generate_image.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── explicit steps / guidance / negative_prompt (MCP surface) ──────────────────


def test_resolve_inference_params_guidance_free_spec_blocks_all_guidance():
    """Guidance-free specs (turbo variants) ignore explicit and preset
    guidance alike — mflux would silently coerce it to 0.0, so honoring
    the kwarg would make a caller believe CFG applied when it did not."""
    spec = get_model_spec("zimage-turbo")  # guidance=None (turbo)
    assert resolve_inference_params(spec, guidance=2.0, preset={"steps": 4}) == (4, None)
    _, preset_blocked = resolve_inference_params(
        spec, preset={"steps": None, "guidance": 3.5}
    )
    assert preset_blocked is None


def test_generate_image_explicit_steps_and_guidance_override_preset(tmp_path):
    """MCP callers can pin steps/guidance per call, beating the preset."""
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": 5, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="zimage",
        steps=7,
        guidance=2.5,
        model=mock_model,
    )

    kwargs = mock_model.generate_image.call_args.kwargs
    assert kwargs["num_inference_steps"] == 7
    assert kwargs["guidance"] == 2.5


def test_generate_image_negative_prompt_passthrough_including_empty(tmp_path):
    """An explicit empty string is forwarded verbatim (mflux substitutes a
    space for it, so it is behaviorally identical to omission — but the
    caller's explicit value must not be second-guessed here)."""
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": None, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="zimage",
        negative_prompt="",
        model=mock_model,
    )

    assert mock_model.generate_image.call_args.kwargs["negative_prompt"] == ""


def test_generate_image_negative_prompt_absent_not_forwarded(tmp_path):
    """No negative_prompt kwarg must reach models that were not asked for one."""
    from fluxgen.generator import generate_image

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.image = MagicMock()
    mock_model.generate_image.return_value = mock_result

    generate_image(
        prompt="a fox",
        preset={"steps": None, "guidance": None, "quantize": 8},
        seed=1,
        output=str(tmp_path / "out.png"),
        width=64,
        height=64,
        style="none",
        model_name="zimage",
        model=mock_model,
    )

    assert "negative_prompt" not in mock_model.generate_image.call_args.kwargs


def test_generate_image_rejects_negative_prompt_for_unsupported_model(tmp_path):
    """Flux.2 Klein's generate_image has no negative_prompt kwarg; the
    generator must reject up front with a clear error, not let a
    TypeError escape from inside mflux."""
    from fluxgen.generator import generate_image

    with pytest.raises(ValueError, match="does not support negative_prompt"):
        generate_image(
            prompt="a fox",
            preset={"steps": None, "guidance": None, "quantize": 8},
            seed=1,
            output=str(tmp_path / "out.png"),
            width=64,
            height=64,
            style="none",
            model_name="flux2-klein4b",
            negative_prompt="blurry",
            model=MagicMock(),
        )
