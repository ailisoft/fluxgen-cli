"""MCP server entry point.

Wires together:
  - `MCPSettings` (loaded from `.fluxgen.toml`)
  - `AuditLog` (JSONL append-only, 0600)
  - `ConcurrencyGate` (max concurrent + queue depth)
  - Pause sentinel check (per-call, inside tool)
  - Two tools: `generate_image`, `edit_image`
  - PID file write + atexit cleanup

Transport: stdio only (per design). The MCP server speaks MCP over
stdin/stdout; logs go to stderr so they don't pollute the protocol
stream.

Per-call timeout: enforced via `asyncio.wait_for` (Python 3.10+
compatible — `asyncio.timeout` is 3.11+). The thread itself does
not get killed mid-inference (mflux doesn't support that
gracefully) but the wrapper returns `E_TIMEOUT` once the wall-clock
budget elapses so the agent isn't left waiting forever.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from fluxgen_mcp.config import MCPSettings, load_mcp_settings
from fluxgen_mcp.errors import E_BUSY, E_INTERNAL, E_TIMEOUT, MCPError
from fluxgen_mcp.safety import (
    AuditLog,
    ConcurrencyGate,
    make_audit_record,
)
from fluxgen_mcp.tools.edit import edit_image_tool
from fluxgen_mcp.tools.generate import generate_image_tool

logger = logging.getLogger("fluxgen-mcp")


def _setup_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger("fluxgen-mcp")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


def _write_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))
    os.chmod(path, 0o600)


def _remove_pid_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _session_agent_id(ctx: Context | None) -> str | None:
    """Best-effort extraction of a session/agent identifier from the
    MCP `Context`. Returns None if any attribute lookup fails —
    never raises (audit logging must never crash the tool).
    """
    if ctx is None:
        return None
    try:
        session = ctx.session
        if session is None:
            return None
        # MCP 2.0 canonical field is `client_info`; older servers
        # exposed `clientInfo`. Try both.
        info = getattr(session, "client_info", None) or getattr(session, "clientInfo", None)
        if info is None:
            return None
        return getattr(info, "name", None) or getattr(info, "title", None)
    except Exception:  # pragma: no cover - defensive
        return None


def _join_workers(timeout_s: float = 5.0) -> None:
    """Sweep non-main live threads at shutdown.

    Worker threads spawned by `asyncio.to_thread` (e.g. the
    diffusion pipeline) are NOT cancelled by `asyncio.wait_for`
    when the timeout fires — they keep running until the
    inference call returns. Under repeated timeouts, zombie
    threads pile up; joining them at shutdown reclaims the
    resources.

    Resolves `main_thread` lazily so the function is safe to
    register via `atexit` before any server is constructed
    (e.g. to handle partial-init failure).
    """
    try:
        main = threading.main_thread()
    except RuntimeError:
        main = threading.current_thread()
    for t in threading.enumerate():
        if t is main or not t.is_alive():
            continue
        t.join(timeout=timeout_s)
        if t.is_alive():
            logger.warning(
                "worker thread %s did not exit within %ss; abandoning",
                t.name, timeout_s,
            )


def build_server(settings: MCPSettings) -> MCPServer:
    """Construct an `MCPServer` with tools registered.

    Returns a configured server. The caller is responsible for
    calling `run`.
    """
    audit = AuditLog(settings.audit_log_path)
    gate = ConcurrencyGate(
        max_concurrent=settings.max_concurrent_jobs,
        max_queue_depth=settings.max_queue_depth,
    )

    server = MCPServer(
        name="fluxgen-mcp",
        instructions=(
            "Local image generation and editing using fluxgen-cli. "
            "All output paths are confined to a sandbox root; absolute "
            "paths and `..` are rejected. Tools may take seconds to "
            "minutes depending on resolution and steps."
        ),
    )

    async def _with_safety(
        tool_name: str,
        prompt: str,
        body,
        ctx: Context | None,
        negative_prompt: str | None = None,
    ):
        """Run a tool body inside the safety envelope.

        Steps:
          1. Acquire the concurrency gate. `E_BUSY` is converted to
             `ToolError` here so a full queue surfaces as a clean
             tool error rather than propagating as a JSON-RPC
             protocol exception.
          2. Start the wall-clock timer.
          3. Run the tool body under `asyncio.wait_for` (Python
             3.10+ compatible) — `E_TIMEOUT` on overrun.
          4. Catch `MCPError` from the tool body, extract the
             `seed`/`model`/`output_path` metadata it carried,
             and convert to `ToolError` for the MCP transport.
          5. Write the audit record (success or error).
          6. Release the gate.
        """
        agent_id = _session_agent_id(ctx)

        # Acquire gate; surface E_BUSY as ToolError so the agent sees
        # a structured tool error, not an unhandled exception.
        try:
            await gate.acquire()
        except MCPError as exc:
            audit.write(
                make_audit_record(
                    tool=tool_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=None,
                    seed=None,
                    started_at=time.perf_counter(),
                    ended_at=time.perf_counter(),
                    result="error",
                    error_code=exc.code,
                    agent_id=agent_id,
                )
            )
            raise ToolError(f"{exc.code}: {exc.message}") from exc

        started = time.perf_counter()
        acquired = True
        try:
            try:
                result = await asyncio.wait_for(
                    body(),
                    timeout=settings.per_call_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                audit.write(
                    make_audit_record(
                        tool=tool_name,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=None,
                        seed=None,
                        started_at=started,
                        ended_at=time.perf_counter(),
                        result="error",
                        error_code=E_TIMEOUT,
                        agent_id=agent_id,
                    )
                )
                raise ToolError(
                    f"{E_TIMEOUT}: per_call_timeout_s ({settings.per_call_timeout_s}s) exceeded"
                ) from exc
            except ToolError:
                # MCP-native error from a deeper seam — pass through
                # unchanged. Avoid the bare-Exception clause below
                # re-wrapping it as E_INTERNAL.
                audit.write(
                    make_audit_record(
                        tool=tool_name,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=None,
                        seed=None,
                        started_at=started,
                        ended_at=time.perf_counter(),
                        result="error",
                        error_code="TOOL_ERROR",
                        agent_id=agent_id,
                    )
                )
                raise
            except MCPError as exc:
                # Tool-layer error. The MCPError may carry seed /
                # model / output_path instance attributes — use
                # them for the audit record.
                meta = exc.audit_meta()
                audit.write(
                    make_audit_record(
                        tool=tool_name,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=meta.get("model"),
                        seed=meta.get("seed"),
                        started_at=started,
                        ended_at=time.perf_counter(),
                        result="error",
                        error_code=exc.code,
                        agent_id=agent_id,
                        output_path=meta.get("output_path"),
                    )
                )
                raise ToolError(f"{exc.code}: {exc.message}") from exc
            except Exception as exc:
                audit.write(
                    make_audit_record(
                        tool=tool_name,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=None,
                        seed=None,
                        started_at=started,
                        ended_at=time.perf_counter(),
                        result="error",
                        error_code=E_INTERNAL,
                        agent_id=agent_id,
                    )
                )
                logger.exception("tool %s raised unexpected error", tool_name)
                raise ToolError(f"{E_INTERNAL}: unexpected error: {exc}") from exc

            # Success: build audit from the result dict.
            audit.write(
                make_audit_record(
                    tool=tool_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=result.get("model"),
                    seed=result.get("seed"),
                    started_at=started,
                    ended_at=time.perf_counter(),
                    result="ok",
                    error_code=None,
                    agent_id=agent_id,
                    output_path=result.get("path"),
                    input_paths=result.get("input_paths"),
                )
            )
            return result
        finally:
            if acquired:
                try:
                    await gate.release()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("gate.release failed")

    @server.tool(
        name="generate_image",
        description=(
            "Generate an image from a text prompt. Output is written under "
            "the configured sandbox root."
        ),
    )
    async def generate_image(
        prompt: str,
        model: str | None = None,
        preset: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        style: str | None = None,
        init_image_path: str | None = None,
        strength: float | None = 0.4,
        output_subdir: str = "default",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        async def _body() -> dict[str, Any]:
            return await generate_image_tool(
                settings=settings,
                prompt=prompt,
                model=model,
                preset=preset,
                steps=steps,
                guidance=guidance,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                style=style,
                init_image_path=init_image_path,
                strength=strength,
                output_subdir=output_subdir,
            )

        return await _with_safety(
            "generate_image", prompt, _body, ctx, negative_prompt=negative_prompt
        )

    @server.tool(
        name="edit_image",
        description=(
            "Edit one or more input images using a natural-language "
            "instruction. Supports multi-image input via flux2-klein-edit."
        ),
    )
    async def edit_image(
        input_paths: list[str],
        prompt: str,
        model: str | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        width: int | None = None,
        height: int | None = None,
        output_subdir: str = "default",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        async def _body() -> dict[str, Any]:
            return await edit_image_tool(
                settings=settings,
                input_paths=input_paths,
                prompt=prompt,
                model=model,
                seed=seed,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                output_subdir=output_subdir,
            )

        return await _with_safety("edit_image", prompt, _body, ctx)

    # Expose for tests; not part of the public API.
    server._concurrency_gate = gate  # noqa: SLF001

    return server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="fluxgen-mcp",
        description="MCP server for fluxgen-cli (stdio transport).",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved MCP settings and exit.",
    )
    p.add_argument(
        "--audit-log",
        default=None,
        help="Override the audit log path (CLI arg beats config).",
    )
    return p.parse_args(argv)


def _apply_cli_overrides(
    settings: MCPSettings, audit_log_override: str | None
) -> MCPSettings:
    """Return a copy of `settings` with the `--audit-log` override applied.

    Uses `dataclasses.replace` rather than constructing a fresh
    instance from `__dict__`, so adding a new field to `MCPSettings`
    later doesn't silently drop it here.
    """
    if not audit_log_override:
        return settings
    return dataclasses.replace(
        settings,
        audit_log_path=str(Path(audit_log_override).expanduser().resolve()),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging()

    settings = load_mcp_settings().expand_paths()
    settings = _apply_cli_overrides(settings, args.audit_log)

    if args.print_config:
        for k, v in sorted(settings.__dict__.items()):
            print(f"{k}={v}")
        return 0

    pid_path = Path(settings.pid_file_path)
    _write_pid_file(pid_path)
    # atexit removes the PID file on any normal exit path (SIGTERM,
    # SIGINT, return from `server.run`, uncaught exception). The
    # default signal dispositions terminate the process which
    # triggers atexit; we deliberately do NOT install custom signal
    # handlers because logging + signal.signal from inside a handler
    # is not async-signal-safe and can deadlock on stdio writes.
    atexit.register(_remove_pid_file, pid_path)
    # Reap leaked worker threads (timed-out `asyncio.to_thread`
    # workers, mostly) at shutdown. Registered before `build_server`
    # so partial-init failures still trigger the cleanup. Uses the
    # module-level `_join_workers` so there's no dangling closure.
    atexit.register(_join_workers)

    server = build_server(settings)

    logger.info(
        "fluxgen-mcp starting (transport=stdio, sandbox=%s, models=%s)",
        settings.output_root,
        settings.allowed_generation_models,
    )

    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())