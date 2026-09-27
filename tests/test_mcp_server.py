"""Tests for the MCP server's safety envelope.

Exercises `_with_safety` indirectly through the public
`build_server(...)` route — but without launching stdio transport.
We call the registered tool functions directly, which is how
`_with_safety` is wired internally.

Coverage:
  - E_BUSY surfaces as `ToolError` (not raw MCPError)
  - timeout → ToolError(E_TIMEOUT) + audit record
  - tool-layer MCPError → ToolError + audit with seed/model/output_path
  - bare Exception → ToolError(E_INTERNAL)
  - ToolError from inner body passes through
  - success path → audit record has seed/model/path
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import E_BUSY, E_INTERNAL, E_TIMEOUT, MCPError
from fluxgen_mcp.server import build_server


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
        allowed_generation_models=("zimage-turbo",),
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


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def mock_generate(monkeypatch):
    def fake_generate(*, prompt, preset, seed, output, width, height,
                      style, init_image, strength, model_name, **_):
        from PIL import Image

        Image.new("RGB", (1, 1), (255, 255, 255)).save(output)

    monkeypatch.setattr("fluxgen_mcp.tools.generate.generate_image", fake_generate)


def _get_tool(server, name: str):
    """Pull the registered tool coroutine out of the server."""
    tools = server._tool_manager._tools  # noqa: SLF001 — testing seam
    tool = tools[name]
    return tool.fn


pytestmark = pytest.mark.asyncio


async def test_success_writes_audit_record_with_seed_and_path(tmp_path: Path, mock_generate):
    s = _settings(tmp_path)
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    result = await fn(prompt="a cat", seed=42, output_subdir="default")
    assert result["path"].endswith(".png")

    records = _read_audit(Path(s.audit_log_path))
    assert len(records) == 1
    rec = records[0]
    assert rec["tool"] == "generate_image"
    assert rec["result"] == "ok"
    assert rec["seed"] == 42
    assert rec["model"] == "zimage-turbo"
    assert rec["output_path"] == result["path"]
    assert rec["error_code"] is None
    # generate_image has no input_paths; the field is omitted.
    assert "input_paths" not in rec
    # No negative_prompt supplied; the key is omitted.
    assert "negative_prompt" not in rec


async def test_negative_prompt_reaches_audit_record(tmp_path: Path, mock_generate):
    """Server-level: a supplied negative_prompt lands in the audit record
    through _with_safety (model-facing content audited like the prompt)."""
    s = _settings(tmp_path, allowed_generation_models=("zimage",))
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    await fn(prompt="a cat", model="zimage", negative_prompt="blurry",
             output_subdir="default")

    records = _read_audit(Path(s.audit_log_path))
    assert len(records) == 1
    assert records[0]["negative_prompt"] == "blurry"


async def test_edit_success_writes_audit_record_with_input_paths(
    tmp_path: Path, monkeypatch,
):
    """`edit_image` must populate `input_paths` in the audit record
    so traceability is preserved when the stem prefix is dropped
    from the output filename.
    """
    from PIL import Image
    from unittest.mock import MagicMock

    s = _settings(tmp_path)
    inp = tmp_path / "input.png"
    Image.new("RGB", (50, 50), (255, 0, 0)).save(inp)

    # Inline ImageEditor mock (avoids cross-file fixture dependency).
    def fake_ctor(*, model_name, quantize=None):
        inst = MagicMock()
        inst.model_name = model_name
        inst.quantize = quantize

        def fake_edit(*, image_paths, prompt, output_path, **_):
            Image.new("RGB", (1, 1), (255, 255, 255)).save(output_path)

        inst.load = MagicMock()
        inst.edit = MagicMock(side_effect=fake_edit)
        return inst

    monkeypatch.setattr("fluxgen_mcp.tools.edit.ImageEditor", fake_ctor)

    server = build_server(s)
    fn = _get_tool(server, "edit_image")

    result = await fn(
        input_paths=[str(inp)], prompt="x", output_subdir="default",
    )

    records = _read_audit(Path(s.audit_log_path))
    assert len(records) == 1
    rec = records[0]
    assert rec["tool"] == "edit_image"
    assert rec["result"] == "ok"
    assert rec["input_paths"] == [str(inp.resolve())]
    assert rec["output_path"] == result["path"]


async def test_busy_surfaces_as_tool_error(tmp_path: Path, mock_generate):
    """Full queue → `ToolError` with `E_BUSY` (not a raw MCPError
    that would crash the MCP transport).
    """
    s = _settings(tmp_path, max_concurrent_jobs=1, max_queue_depth=0)
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    # Hold the only slot with a blocking tool body. `slow_generate`
    # runs in a worker thread (via `asyncio.to_thread`), so it must
    # use `threading.Event` rather than `asyncio.Event`.
    import threading
    started = threading.Event()

    def slow_generate(*, prompt, preset, seed, output, width, height,
                      style, init_image, strength, model_name, **_):
        from PIL import Image
        Image.new("RGB", (1, 1), (255, 255, 255)).save(output)
        started.wait()  # blocks the worker thread

    import fluxgen_mcp.tools.generate as gen_mod
    original = gen_mod.generate_image
    gen_mod.generate_image = slow_generate
    try:
        running = asyncio.create_task(fn(prompt="a cat", seed=1))
        # Wait for the running task to acquire the gate. We probe
        # indirectly — there's no public stats accessor — by waiting
        # for the audit log to receive the success record (the
        # block happens after `generate_image` returns, so success
        # is only logged once the slot is held).
        # Faster: spin until `gate._in_flight >= 1`.
        for _ in range(1000):
            await asyncio.sleep(0)
            if server._concurrency_gate._in_flight >= 1:  # noqa: SLF001
                break

        # Second call should be rejected (queue depth 0).
        with pytest.raises(ToolError) as exc:
            await asyncio.wait_for(
                fn(prompt="a cat", seed=2), timeout=0.5,
            )
        assert E_BUSY in str(exc.value)
    finally:
        started.set()
        await running
        gen_mod.generate_image = original

    # Audit record for the rejected call should be present and
    # tagged E_BUSY.
    records = _read_audit(Path(s.audit_log_path))
    busy_records = [r for r in records if r.get("error_code") == E_BUSY]
    assert len(busy_records) >= 1


async def test_timeout_surfaces_as_tool_error(tmp_path: Path, mock_generate):
    s = _settings(tmp_path, per_call_timeout_s=0.05)
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    # The tool layer's generate_image fake completes instantly; we
    # instead wrap by injecting a slow `_with_safety` body via the
    # body callable the server builds. Easiest path: make the tool
    # layer hang in `asyncio.to_thread`.
    import fluxgen_mcp.tools.generate as gen_mod

    def slow(*, prompt, preset, seed, output, width, height, style, init_image,
             strength, model_name, **_):
        import time
        time.sleep(0.5)  # exceeds the 0.05s timeout
        from PIL import Image
        Image.new("RGB", (1, 1), (255, 255, 255)).save(output)

    original = gen_mod.generate_image
    gen_mod.generate_image = slow
    try:
        with pytest.raises(ToolError) as exc:
            await fn(prompt="a cat", seed=1)
        assert E_TIMEOUT in str(exc.value)
    finally:
        gen_mod.generate_image = original

    records = _read_audit(Path(s.audit_log_path))
    timeout_records = [r for r in records if r.get("error_code") == E_TIMEOUT]
    assert len(timeout_records) == 1


async def test_tool_error_passes_through(tmp_path: Path, mock_generate):
    """If the tool body raises `ToolError`, `_with_safety` should
    NOT re-wrap it as E_INTERNAL. It records an audit entry and
    re-raises the original.
    """
    s = _settings(tmp_path)
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    # Inject a generate_image that raises ToolError.
    import fluxgen_mcp.tools.generate as gen_mod

    def fake(*, prompt, preset, seed, output, width, height, style,
             init_image, strength, model_name, **_):
        raise ToolError("native tool error from inner seam")

    original = gen_mod.generate_image
    gen_mod.generate_image = fake
    try:
        with pytest.raises(ToolError) as exc:
            await fn(prompt="a cat", seed=1)
        assert "native tool error" in str(exc.value)
        # Critical: must NOT be wrapped as E_INTERNAL.
        assert E_INTERNAL not in str(exc.value)
    finally:
        gen_mod.generate_image = original


async def test_mcperror_from_tool_layer_populates_audit(tmp_path: Path, mock_generate):
    """`MCPError` raised by the tool layer carries seed + model +
    output_path metadata on its instance. The server's audit
    record must use that metadata, not the empty defaults from
    the previous design.
    """
    s = _settings(tmp_path)
    server = build_server(s)
    fn = _get_tool(server, "generate_image")

    import fluxgen_mcp.tools.generate as gen_mod

    def fake(*, prompt, preset, seed, output, width, height, style,
             init_image, strength, model_name, **_):
        raise MCPError(
            "E_BAD_ARG",
            "deliberate test failure",
            seed=999,
            model="zimage-turbo",
            output_path=None,
        )

    original = gen_mod.generate_image
    gen_mod.generate_image = fake
    try:
        with pytest.raises(ToolError) as exc:
            await fn(prompt="a cat", seed=999)
        assert "E_BAD_ARG" in str(exc.value)
    finally:
        gen_mod.generate_image = original

    records = _read_audit(Path(s.audit_log_path))
    assert len(records) == 1
    rec = records[0]
    assert rec["result"] == "error"
    assert rec["error_code"] == "E_BAD_ARG"
    assert rec["seed"] == 999  # not None
    assert rec["model"] == "zimage-turbo"  # not None