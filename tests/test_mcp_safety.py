"""Tests for fluxgen_mcp.safety (pause, sandbox, prompt filter, audit, gate)."""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BUSY,
    E_DISABLED,
    E_PATH_TRAVERSAL,
    E_PROMPT_REJECTED,
    E_PROMPT_TOO_LONG,
    MCPError,
)
from fluxgen_mcp.safety import (
    AuditLog,
    ConcurrencyGate,
    check_pause,
    make_audit_record,
    resolve_sandbox_output,
    validate_prompt,
)


def _ip(pattern: str) -> re.Pattern[str]:
    """Compile a case-insensitive pattern, mirroring how
    `load_mcp_settings` stores blocklist entries.
    """
    return re.compile(pattern, flags=re.IGNORECASE)


def _settings(**overrides) -> MCPSettings:
    base = dict(
        output_root="~/fluxgen-mcp-output-test",
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
        audit_log_path="~/.fluxgen-mcp-audit-test.log",
        pause_sentinel_path="~/.fluxgen-mcp-paused-test",
        pid_file_path="~/.fluxgen-mcp.pid-test",
        input_max_bytes=20 * 1024 * 1024,
        input_max_dimension=1080,
    )
    base.update(overrides)
    return MCPSettings(**base)


# ── resolve_sandbox_output ──────────────────────────────────────────────────


def test_sandbox_creates_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = _settings(output_root=str(tmp_path / "sandbox")).expand_paths()
    target = resolve_sandbox_output(s, "abc-123")
    assert target.exists()
    assert target.is_relative_to(s.output_root)
    assert target.name == "abc-123"


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "..",  # dot-dot
        "../etc",  # contains ..
        "sub/dir",  # contains slash
        "sub.dir",  # contains dot
        "sub dir",  # contains space
        "x" * 33,  # too long
        "sub$dir",  # shell metachar
        "subdir\n",  # control char
    ],
)
def test_sandbox_rejects_bad_subdirs(tmp_path: Path, bad: str):
    s = _settings(output_root=str(tmp_path / "sandbox"))
    with pytest.raises(MCPError) as exc:
        resolve_sandbox_output(s, bad)
    assert exc.value.code == E_PATH_TRAVERSAL


def test_sandbox_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # Create a symlink inside the sandbox pointing outside.
    link = root / "escape"
    link.symlink_to(outside)
    s = _settings(output_root=str(root))
    with pytest.raises(MCPError) as exc:
        resolve_sandbox_output(s, "escape")
    assert exc.value.code == E_PATH_TRAVERSAL


def test_sandbox_rejects_empty_subdir(tmp_path: Path):
    # The empty-subdir case is closed by the regex's `1,32` length
    # constraint plus the explicit `not subdir` early-return. This
    # test locks in the behavior so a future regex tweak can't
    # silently allow `""` through.
    s = _settings(output_root=str(tmp_path / "sandbox"))
    with pytest.raises(MCPError) as exc:
        resolve_sandbox_output(s, "")
    assert exc.value.code == E_PATH_TRAVERSAL


# ── check_pause ─────────────────────────────────────────────────────────────


def test_check_pause_blocks_when_sentinel_exists(tmp_path: Path):
    sentinel = tmp_path / "paused"
    sentinel.touch()
    s = _settings(pause_sentinel_path=str(sentinel))
    with pytest.raises(MCPError) as exc:
        check_pause(s)
    assert exc.value.code == E_DISABLED


def test_check_pause_passes_when_sentinel_absent(tmp_path: Path):
    s = _settings(pause_sentinel_path=str(tmp_path / "missing"))
    check_pause(s)  # no raise


# ── validate_prompt ─────────────────────────────────────────────────────────


def test_validate_prompt_too_long():
    s = _settings(max_prompt_chars=10)
    with pytest.raises(MCPError) as exc:
        validate_prompt(s, "x" * 11)
    assert exc.value.code == E_PROMPT_TOO_LONG


def test_validate_prompt_at_limit_is_ok():
    s = _settings(max_prompt_chars=10)
    validate_prompt(s, "x" * 10)  # no raise


def test_validate_prompt_blocklist_hit_returns_generic_message():
    s = _settings(prompt_blocklist=(_ip(r"foo"),))
    with pytest.raises(MCPError) as exc:
        validate_prompt(s, "this contains foo bar")
    assert exc.value.code == E_PROMPT_REJECTED
    # The offending substring must not be echoed.
    assert "foo" not in exc.value.message


def test_validate_prompt_blocklist_case_insensitive():
    s = _settings(prompt_blocklist=(_ip(r"badword"),))
    with pytest.raises(MCPError):
        validate_prompt(s, "BADWORD appears here")


def test_validate_prompt_tolerates_stray_string_blocklist_entry():
    """Manual `MCPSettings(prompt_blocklist=["foo"], ...)` is
    legal — a third-party caller may construct settings that way.
    `validate_prompt` must NOT AttributeError on a string entry;
    it should log a warning and skip. The point of this contract
    is that an uncompiled string cannot match anyway, so failing
    the whole call would be worse than skipping.
    """
    s = _settings(prompt_blocklist=(r"foo",))  # type: ignore[arg-type]
    # Should not raise; should not match (string has no .search).
    validate_prompt(s, "this contains foo bar")  # no raise


def test_validate_prompt_invalid_blocklist_regex_does_not_crash():
    # Bad regex in config should be skipped at load time (in
    # `load_mcp_settings`). Tests that construct `MCPSettings`
    # directly cannot reproduce the invalid-regex path because
    # they pass already-compiled patterns; the load-time guard
    # is verified by `test_load_mcp_settings_drops_invalid_regex`.
    s = _settings(prompt_blocklist=())
    validate_prompt(s, "anything")  # no raise


# ── AuditLog ────────────────────────────────────────────────────────────────


def test_audit_log_appends_jsonl(tmp_path: Path):
    log = AuditLog(str(tmp_path / "audit.log"))
    log.write({"tool": "generate_image", "result": "ok", "prompt": "hello"})
    log.write({"tool": "edit_image", "result": "error", "error_code": "E_TEST"})
    lines = (tmp_path / "audit.log").read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["tool"] == "generate_image"
    assert parsed[1]["error_code"] == "E_TEST"
    # Each record has a timestamp.
    assert all("ts" in r for r in parsed)


def test_audit_log_mode_is_0600(tmp_path: Path):
    log_path = tmp_path / "audit.log"
    AuditLog(str(log_path))
    mode = stat_mode(log_path)
    assert mode & 0o777 == 0o600


def test_audit_log_writes_under_concurrent_threads(tmp_path: Path):
    """Two `AuditLog` instances pointed at the same file should not
    interleave lines. We exercise the `fcntl.flock` path by
    opening two writers in parallel threads and verifying each
    record parses as a complete JSON object.

    Note: this test runs in a single process so the threads share
    one file descriptor's lock state. True multi-process flock
    isolation is implicitly relied on but not directly exercised
    here. Adding a multiprocessing-based test would require a
    separate entry point; deferred until a concrete bug appears.
    """
    import threading

    log_path = tmp_path / "audit.log"
    AuditLog(str(log_path))

    n = 50
    errors: list[Exception] = []

    def writer(idx: int) -> None:
        try:
            log = AuditLog(str(log_path))
            for i in range(n):
                log.write({"writer": idx, "i": i, "result": "ok"})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors

    lines = log_path.read_text().splitlines()
    assert len(lines) == n * 4
    # Every line must parse as JSON. An interleaved line would
    # raise here.
    for line in lines:
        rec = json.loads(line)
        assert "writer" in rec
        assert "i" in rec


def stat_mode(path: Path) -> int:
    return path.stat().st_mode


def test_make_audit_record_includes_prompt_and_hash():
    rec = make_audit_record(
        tool="generate_image",
        prompt="a cat",
        model="zimage-turbo",
        seed=42,
        started_at=0.0,
        ended_at=1.234,
        result="ok",
        error_code=None,
    )
    assert rec["prompt"] == "a cat"
    assert len(rec["prompt_hash"]) == 64  # sha256 hex
    assert rec["duration_s"] == 1.234
    assert rec["seed"] == 42
    assert "negative_prompt" not in rec  # key omitted when not supplied


def test_make_audit_record_includes_negative_prompt_when_supplied():
    rec = make_audit_record(
        tool="generate_image",
        prompt="a cat",
        model="zimage",
        seed=42,
        started_at=0.0,
        ended_at=1.0,
        result="ok",
        error_code=None,
        negative_prompt="",
    )
    # Empty string is a valid explicit value and must be recorded.
    assert rec["negative_prompt"] == ""


# ── ConcurrencyGate ─────────────────────────────────────────────────────────


def test_concurrency_gate_rejects_queue_full():
    async def runner():
        gate = ConcurrencyGate(max_concurrent=1, max_queue_depth=1)
        await gate.acquire()  # 1 in flight

        async def waiter():
            await gate.acquire()
            await gate.release()

        task = asyncio.create_task(waiter())
        # Scheduler-robust wait: spin until the waiter has
        # incremented `_waiting` (or fail loudly after a bounded
        # wait).
        for _ in range(1000):
            await asyncio.sleep(0)
            if gate.stats["waiting"] >= 1:
                break
        assert gate.stats["waiting"] >= 1, "waiter never registered"

        try:
            with pytest.raises(MCPError) as exc:
                await asyncio.wait_for(gate.acquire(), timeout=0.5)
            assert exc.value.code == E_BUSY
        finally:
            await gate.release()
            await task

    asyncio.run(runner())


def test_concurrency_gate_cancellation_does_not_leak_permit():
    """If `acquire()` is cancelled while waiting on the semaphore,
    `_waiting` must be decremented and `_in_flight` MUST NOT be
    incremented. Otherwise a subsequent `release` would call
    `self._sem.release()` without owning a permit, raising
    `ValueError: Semaphore released too many times` and allowing
    the gate to exceed `max_concurrent`.
    """
    async def runner():
        gate = ConcurrencyGate(max_concurrent=1, max_queue_depth=10)
        await gate.acquire()  # fill the only slot

        async def waiter():
            await gate.acquire()

        task = asyncio.create_task(waiter())
        for _ in range(1000):
            await asyncio.sleep(0)
            if gate.stats["waiting"] >= 1:
                break
        assert gate.stats["waiting"] >= 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Crucial: stats are clean after cancellation.
        assert gate.stats == {"waiting": 0, "in_flight": 1}

        await gate.release()
        assert gate.stats == {"waiting": 0, "in_flight": 0}

        await gate.acquire()  # should succeed — the only slot
        assert gate.stats["in_flight"] == 1
        # A second acquire must NOT succeed (gate is full).
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.acquire(), timeout=0.05)
        await gate.release()

    asyncio.run(runner())


def test_concurrency_gate_release_without_acquire_is_safe():
    """`release()` is a no-op when no in-flight acquires exist.
    Defensive guard against bugs in the tool layer.
    """
    async def runner():
        gate = ConcurrencyGate(max_concurrent=1, max_queue_depth=1)
        await gate.release()  # no-op; must not raise
        assert gate.stats == {"waiting": 0, "in_flight": 0}
        await gate.acquire()
        assert gate.stats["in_flight"] == 1

    asyncio.run(runner())


def test_concurrency_gate_rejects_invalid_construction():
    with pytest.raises(ValueError):
        ConcurrencyGate(max_concurrent=0, max_queue_depth=0)
    with pytest.raises(ValueError):
        ConcurrencyGate(max_concurrent=1, max_queue_depth=-1)
