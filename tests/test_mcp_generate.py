"""Tests for the generate_image tool wrapper.

The underlying `fluxgen.generator.generate_image` is mocked at the
import seam so the tests don't load mflux.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BAD_ARG,
    E_INVALID_INPUT_IMAGE,
    E_PATH_TRAVERSAL,
    E_PROMPT_REJECTED,
    E_PROMPT_TOO_LONG,
    MCPError,
)
from fluxgen_mcp.tools.generate import generate_image_tool


def _settings(tmp_path: Path, **overrides) -> MCPSettings:
    base = dict(
        output_root=str(tmp_path / "out"),
        max_width=1920,
        max_height=1920,
        max_steps=50,
        max_prompt_chars=2000,
        max_concurrent_jobs=1,
        max_queue_depth=4,
        per_call_timeout_s=600.0,
        allowed_generation_models=("zimage-turbo", "zimage"),
        allowed_edit_models=("flux2-klein-edit",),
        prompt_blocklist=(),
        audit_log_path=str(tmp_path / "audit.log"),
        pause_sentinel_path=str(tmp_path / "paused"),
        pid_file_path=str(tmp_path / "pid"),
        input_max_bytes=20_000_000,
        input_max_dimension=1080,
    )
    base.update(overrides)
    return MCPSettings(**base)


@pytest.fixture
def mock_generate(monkeypatch):
    """Replace fluxgen.generator.generate_image with a fake.

    Records calls and writes a 1x1 PNG so downstream validation
    (file existence, format) doesn't fail.
    """
    calls = []

    def fake_generate(*, prompt, preset, seed, output, width, height,
                      style, init_image, strength, model_name, **extra):
        calls.append({
            "prompt": prompt,
            "preset": preset,
            "seed": seed,
            "output": output,
            "width": width,
            "height": height,
            "style": style,
            "init_image": init_image,
            "strength": strength,
            "model_name": model_name,
            **extra,
        })
        Image.new("RGB", (1, 1), (255, 255, 255)).save(output)

    monkeypatch.setattr("fluxgen_mcp.tools.generate.generate_image", fake_generate)
    return calls


pytestmark = pytest.mark.asyncio


async def test_generate_basic(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a cat",
        model=None,
        preset=None,
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="cats",
    )
    assert result["model"] == "zimage-turbo"
    assert result["width"] == 512
    assert result["height"] == 512
    assert "cats" in result["path"]
    assert Path(result["path"]).exists()
    assert len(mock_generate) == 1
    assert mock_generate[0]["prompt"] == "a cat"
    assert mock_generate[0]["strength"] == 0.4  # default


async def test_generate_with_seed_and_dimensions(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a fox",
        model="zimage",
        preset="quality",
        width=1024,
        height=1024,
        seed=42,
        style="cinematic",
        init_image_path=None,
        strength=0.7,
        output_subdir="default",
    )
    assert result["seed"] == 42
    assert mock_generate[0]["seed"] == 42
    assert mock_generate[0]["model_name"] == "zimage"
    assert mock_generate[0]["width"] == 1024
    assert mock_generate[0]["height"] == 1024
    assert mock_generate[0]["strength"] == 0.7


async def test_generate_rejects_model_not_in_whitelist(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="nonexistent-model",
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_rejects_dimensions_over_cap(tmp_path: Path, mock_generate):
    s = _settings(tmp_path, max_width=800)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=1024,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_rejects_prompt_too_long(tmp_path: Path, mock_generate):
    s = _settings(tmp_path, max_prompt_chars=10)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="x" * 100,
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_PROMPT_TOO_LONG


async def test_generate_rejects_path_traversal(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="../../etc",
        )
    assert exc.value.code == E_PATH_TRAVERSAL


async def test_generate_respects_pause(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    sentinel = tmp_path / "paused"
    sentinel.touch()
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == "E_DISABLED"
    assert len(mock_generate) == 0


async def test_generate_attach_meta_to_mcperror(tmp_path: Path, monkeypatch, mock_generate):
    """Tool-layer errors must carry seed + model + output_path so the
    server's audit writer can populate them."""
    from fluxgen.exceptions import FluxgenError

    def boom(*, prompt, preset, seed, output, width, height, style,
             init_image, strength, model_name, **kwargs):
        raise FluxgenError("model load failed")

    monkeypatch.setattr("fluxgen_mcp.tools.generate.generate_image", boom)
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=99,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.seed == 99
    assert exc.value.model == "zimage-turbo"
    assert exc.value.output_path is None


async def test_generate_maps_filenotfounderror_init(tmp_path: Path, mock_generate):
    """If the init_image_path does not exist, the tool layer raises
    `E_INVALID_INPUT_IMAGE` (not `E_INTERNAL`) — the validation.py
    short-circuit fires before `validate_image_file`.
    """
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=str(tmp_path / "missing.png"),
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_INVALID_INPUT_IMAGE

async def test_generate_steps_override_preset(tmp_path: Path, mock_generate):
    """Explicit steps beat the preset value and flow through to the generator."""
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a cat",
        model=None,
        preset="quality",
        steps=40,
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["steps"] == 40
    assert result["steps"] == 40


async def test_generate_steps_exceeding_max_steps_rejected(tmp_path: Path):
    """Caller steps above settings.max_steps are a bad argument."""
    s = _settings(tmp_path, max_steps=50)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            steps=51,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG
    assert "steps must be in [1, 50]" in str(exc.value)


async def test_generate_steps_below_one_rejected(tmp_path: Path):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            steps=0,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_result_reports_effective_preset_steps(
    tmp_path: Path, mock_generate
):
    """Without an override, the result's steps reflect the chosen preset."""
    s = _settings(tmp_path)
    result = await generate_image_tool(
        settings=s,
        prompt="a cat",
        model=None,
        preset="standard",
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["steps"] is None  # unset → forwarded as None
    assert result["steps"] == 9  # standard preset


async def test_generate_guidance_passthrough(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    await generate_image_tool(
        settings=s,
        prompt="a cat",
        model="zimage",
        preset=None,
        guidance=2.5,
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["guidance"] == 2.5


async def test_generate_nonpositive_guidance_rejected(tmp_path: Path):
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            guidance=0,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_negative_prompt_passthrough(tmp_path: Path, mock_generate):
    """A negative prompt (including empty) reaches the generator untouched.

    Uses `zimage` — the default zimage-turbo is guidance-free, so its
    sampler never encodes a negative prompt and the tool rejects the
    parameter for it instead of silently no-oping.
    """
    s = _settings(tmp_path, allowed_generation_models=("zimage",))
    await generate_image_tool(
        settings=s,
        prompt="a cat",
        model="zimage",
        preset=None,
        negative_prompt="",
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    await generate_image_tool(
        settings=s,
        prompt="a cat",
        model="zimage",
        preset=None,
        negative_prompt="blurry, low quality",
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["negative_prompt"] == ""
    assert mock_generate[1]["negative_prompt"] == "blurry, low quality"


async def test_generate_negative_prompt_on_guidance_free_default_rejected(
    tmp_path: Path,
):
    """zimage-turbo (the default model) runs guidance-free, so its sampler
    never encodes a negative prompt — reject rather than silently no-op."""
    s = _settings(tmp_path)  # default model zimage-turbo
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            negative_prompt="blurry",
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG
    assert "does not support negative_prompt" in str(exc.value)


async def test_generate_nan_steps_rejected(tmp_path: Path):
    """NaN would slip through naive bound comparisons and crash int()."""
    s = _settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            steps=float("nan"),
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_nan_guidance_rejected(tmp_path: Path):
    s = _settings(tmp_path, allowed_generation_models=("zimage",))
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="zimage",
            preset=None,
            guidance=float("nan"),
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_negative_prompt_absent_not_forwarded(
    tmp_path: Path, mock_generate
):
    """When not requested, the tool forwards None; the generator layer
    (covered in test_generator.py) then omits the kwargs from the model
    call entirely."""
    s = _settings(tmp_path)
    await generate_image_tool(
        settings=s,
        prompt="a cat",
        model=None,
        preset=None,
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["negative_prompt"] is None
    assert mock_generate[0]["guidance"] is None
    assert mock_generate[0]["steps"] is None


async def test_generate_negative_prompt_blocklist(tmp_path: Path):
    """The negative prompt is subject to the same content filter as the prompt."""
    import re

    s = _settings(tmp_path, prompt_blocklist=(re.compile("bomb"),))
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            negative_prompt="a bomb",
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_PROMPT_REJECTED


async def test_generate_guidance_on_guidance_free_model_rejected(tmp_path: Path):
    """Turbo models silently coerce guidance to 0.0 inside mflux; the tool
    must reject instead of letting a caller believe CFG applied."""
    s = _settings(tmp_path)  # zimage-turbo is the default model
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model=None,
            preset=None,
            guidance=2.5,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG
    assert "guidance-free" in str(exc.value)


async def test_generate_negative_prompt_unsupported_model_rejected(tmp_path: Path):
    """Flux.2 Klein has no negative_prompt kwarg in mflux; reject as a bad
    argument instead of a TypeError from inside generation."""
    s = _settings(tmp_path, allowed_generation_models=("flux2-klein4b",))
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="flux2-klein4b",
            preset=None,
            negative_prompt="blurry",
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG
    assert "does not support negative_prompt" in str(exc.value)


async def test_generate_guidance_upper_bound_rejected(tmp_path: Path):
    s = _settings(tmp_path, allowed_generation_models=("zimage",))
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="zimage",
            preset=None,
            guidance=1e9,
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG


async def test_generate_negative_prompt_krea2_default_guidance_rejected(
    tmp_path: Path,
):
    """krea2's spec default guidance is 1.0, where mflux skips negative
    encoding — reject instead of silently no-oping."""
    s = _settings(tmp_path, allowed_generation_models=("krea2",))
    with pytest.raises(MCPError) as exc:
        await generate_image_tool(
            settings=s,
            prompt="a cat",
            model="krea2",
            preset=None,
            negative_prompt="blurry",
            width=None,
            height=None,
            seed=None,
            style=None,
            init_image_path=None,
            strength=None,
            output_subdir="default",
        )
    assert exc.value.code == E_BAD_ARG
    assert "guidance > 1.0" in str(exc.value)


async def test_generate_negative_prompt_with_guidance_gt_1_passes(
    tmp_path: Path, mock_generate
):
    """krea2 + explicit guidance 1.5 + negative_prompt is the valid combo."""
    s = _settings(tmp_path, allowed_generation_models=("krea2",))
    await generate_image_tool(
        settings=s,
        prompt="a cat",
        model="krea2",
        preset=None,
        guidance=1.5,
        negative_prompt="blurry",
        width=None,
        height=None,
        seed=None,
        style=None,
        init_image_path=None,
        strength=None,
        output_subdir="default",
    )
    assert mock_generate[0]["guidance"] == 1.5
    assert mock_generate[0]["negative_prompt"] == "blurry"
