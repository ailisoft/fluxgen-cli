"""Safety primitives shared by the MCP tool layer.

Owns:
  - Pause sentinel check (`check_pause`)
  - Output path sandboxing (`resolve_sandbox_output`)
  - Prompt length + blocklist validation (`validate_prompt`)
  - Audit log writer (`AuditLog`)
  - Concurrency gate with queue depth limit (`ConcurrencyGate`)

Everything is synchronous except `ConcurrencyGate.acquire` /
`release` which are async. The tool layer composes these together.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fluxgen_mcp.config import MCPSettings
from fluxgen_mcp.errors import (
    E_BUSY,
    E_DISABLED,
    E_PATH_TRAVERSAL,
    E_PROMPT_REJECTED,
    E_PROMPT_TOO_LONG,
    MCPError,
)

logger = logging.getLogger("fluxgen-mcp")

# Subdir naming rule. Conservative: alphanumeric + dash + underscore,
# bounded length. Prevents `..`, `/`, shell metacharacters, NULs,
# Unicode look-alikes. `fullmatch` is required (not `match`) so a
# trailing `\n` cannot smuggle through.
SUBPATH_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


def check_pause(settings: MCPSettings) -> None:
    """Raise `MCPError(E_DISABLED)` if the pause sentinel exists."""
    sentinel = Path(settings.pause_sentinel_path)
    if sentinel.exists():
        raise MCPError(E_DISABLED, "server is paused")


def resolve_sandbox_output(settings: MCPSettings, subdir: str) -> Path:
    """Resolve an `output_subdir` to a directory under the sandbox root.

    The directory is created if missing. Caller appends a random
    filename. Containment is enforced via `resolve()` + parent-set
    check against the resolved root (defeats symlink escape).

    Raises:
        MCPError(E_PATH_TRAVERSAL): `subdir` fails the regex, the
            resolved target is outside the sandbox root, or the path
            is a symlink to an unexpected location.
    """
    if not subdir or not SUBPATH_RE.fullmatch(subdir):
        raise MCPError(
            E_PATH_TRAVERSAL,
            "output_subdir must match [a-zA-Z0-9_-]{1,32}",
        )

    root = Path(settings.output_root)
    # resolve() follows symlinks. If the root itself is a symlink that
    # escapes, the resolved path may differ from the literal root.
    root_resolved = root.resolve()
    target = (root_resolved / subdir).resolve()

    if root_resolved == target or root_resolved not in target.parents:
        # The not-in-parents check also rejects `target == root_resolved`
        # (output_subdir must be a subdirectory, not the root itself).
        raise MCPError(
            E_PATH_TRAVERSAL,
            f"output path {target} is outside sandbox root {root_resolved}",
        )

    target.mkdir(parents=True, exist_ok=True)
    return target


def validate_prompt(settings: MCPSettings, prompt: str) -> None:
    """Length + optional regex blocklist check.

    Blocklist patterns are precompiled (case-insensitive) at
    settings-load time, so this function only does the cheap
    `Pattern.search` per pattern. On hit, raises with a generic
    message (the offending substring is NOT echoed back to the
    agent so the agent cannot iterate around the filter by
    scraping its own errors).

    Defensively tolerates a stray string entry in
    `prompt_blocklist`: `MCPSettings` can be constructed manually
    (tests, third-party callers) with raw strings. Such entries
    are silently skipped — they cannot match without a compiled
    pattern, but the caller likely intended the pattern to be
    applied; surfacing an opaque `AttributeError` is worse than
    skipping.
    """
    if not isinstance(prompt, str):
        raise MCPError(E_PROMPT_TOO_LONG, "prompt must be a string")
    if len(prompt) > settings.max_prompt_chars:
        raise MCPError(
            E_PROMPT_TOO_LONG,
            f"prompt exceeds {settings.max_prompt_chars} characters",
        )
    for pattern in settings.prompt_blocklist:
        if not isinstance(pattern, re.Pattern):
            # Avoid logging the full pattern — a caller could
            # accidentally pass a multi-MB string (e.g. a path or
            # prompt). Report length only.
            logger.warning(
                "prompt_blocklist entry is not a compiled Pattern (len=%d); skipping",
                len(pattern) if hasattr(pattern, "__len__") else -1,
            )
            continue
        if pattern.search(prompt):
            raise MCPError(
                E_PROMPT_REJECTED,
                "prompt rejected by content filter",
            )


class AuditLog:
    """Append-only JSONL audit log.

    The file is created atomically with mode 0600 via `os.open`
    so there is no window where it is world-readable. Records are
    flushed after every write so a `kill -9` does not lose the last
    entry. Per-call writes acquire an exclusive `fcntl.flock` so
    two server processes pointed at the same log file (e.g. during
    a dev workflow) do not interleave lines.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic create-with-mode + immediate close. Avoids the
        # `touch`-then-`chmod` race and the `exists()`-then-touch
        # race.
        if not self.path.exists():
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            # Tighten permissions on existing files (e.g. left over
            # from a previous run with a looser umask).
            os.chmod(self.path, 0o600)

    def write(self, record: dict[str, Any]) -> None:
        # Defensive: callers that bypass `make_audit_record` (e.g.
        # the audit-log tests, third-party callers) get a
        # timestamp set here. `make_audit_record` deliberately
        # does NOT set `ts` so this `setdefault` is the single
        # source of truth for the timestamp.
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, default=str) + "\n"
        try:
            fd = os.open(
                str(self.path),
                os.O_WRONLY | os.O_APPEND,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    os.write(fd, line.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning("audit log write failed: %s", exc)


class ConcurrencyGate:
    """Limit concurrent tool calls; reject when queue is full.

    Semantics:
      - At most `max_concurrent` tool bodies run at once.
      - Up to `max_queue_depth` callers can be queued waiting for a
        slot. The (max_queue_depth + 1)-th waiting caller is rejected
        with `E_BUSY` immediately.
      - `acquire` is async; `release` is async.
      - Cancellation-safe: if `await self._sem.acquire()` is
        cancelled mid-wait, `_waiting` is decremented and `_in_flight`
        is NOT incremented (the caller did not actually acquire a
        permit), so a subsequent `release` does not call
        `self._sem.release()` without owning it.
    """

    def __init__(self, max_concurrent: int, max_queue_depth: int):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_queue = max_queue_depth
        self._waiting = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> dict[str, int]:
        return {"waiting": self._waiting, "in_flight": self._in_flight}

    def _sem_value_safe(self) -> int:
        """Read the underlying semaphore's free-slot count without
        going through the public `locked()` API. Returns 0 on any
        AttributeError (e.g. a future Python renames `_value`) —
        falling back to the conservative "treat as full" semantic
        which causes the caller to join the queue rather than race.

        Trade-off: the fallback returns a binary 0/1, not the
        actual count. For `max_concurrent > 1` callers that want
        the exact count, this helper is insufficient; we'd need to
        acquire `self._sem` internally and inspect `_value` under
        that lock. The current call site only needs a binary
        "free or not" so the fallback is acceptable.
        """
        try:
            return self._sem._value  # noqa: SLF001
        except AttributeError:
            # Public fallback: `locked()` returns True iff the
            # semaphore has zero free slots. If we can't see the
            # count, treat the semaphore as full so we don't risk
            # exceeding `max_queue_depth`.
            return 0 if self._sem.locked() else 1

    async def acquire(self) -> None:
        async with self._lock:
            # Reading `_value` here (under `self._lock`) makes the
            # read consistent — no other task can change `_waiting`
            # or `_in_flight` while we hold it. The semaphore's own
            # internal lock governs slot release, so once we see
            # `_value > 0` we know a slot was free at the moment
            # of the read. The actual `sem.acquire()` below can
            # still race: another task may grab the slot between
            # this read and the await, in which case we block and
            # get treated as a waiter. That's a fair FIFO order
            # under asyncio, not a bug.
            free = self._sem_value_safe()
            if free > 0:
                became_waiter = False
            else:
                # All slots in use; caller would block on sem.acquire
                # and become a queued waiter. Enforce queue depth.
                if self._waiting >= self._max_queue:
                    raise MCPError(
                        E_BUSY,
                        f"server at capacity ({self._max_queue} queued); retry later",
                    )
                self._waiting += 1
                became_waiter = True

        try:
            await self._sem.acquire()
        except BaseException:
            # Semaphore was not acquired (cancelled or other
            # BaseException). Decrement `_waiting` only — do NOT
            # increment `_in_flight`, since we don't hold a permit.
            if became_waiter:
                async with self._lock:
                    self._waiting -= 1
            raise
        async with self._lock:
            if became_waiter:
                self._waiting -= 1
            self._in_flight += 1

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight <= 0:
                # Defensive: should never happen if release is paired
                # with a successful acquire. Log and skip.
                logger.warning("release called with no in-flight acquires")
                return
            self._in_flight -= 1
        self._sem.release()


def make_audit_record(
    *,
    tool: str,
    prompt: str,
    model: str | None,
    seed: int | None,
    started_at: float,
    ended_at: float,
    result: str,
    error_code: str | None,
    agent_id: str | None = None,
    output_path: str | None = None,
    input_paths: list[str] | None = None,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Build an audit record dict. Caller writes via `AuditLog.write`.

    `input_paths` is included for `edit_image` (and any future tool
    with multi-file input) so the audit log captures what the
    tool acted on, not just what it produced. `negative_prompt` is
    included for `generate_image` when the caller supplied one: it is
    model-facing content and must be audited on the same terms as the
    prompt.
    """
    import hashlib

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record: dict[str, Any] = {
        "tool": tool,
        "agent_id": agent_id,
        "model": model,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "seed": seed,
        "duration_s": round(ended_at - started_at, 3),
        "result": result,
        "error_code": error_code,
        "output_path": output_path,
    }
    if input_paths is not None:
        record["input_paths"] = list(input_paths)
    if negative_prompt is not None:
        record["negative_prompt"] = negative_prompt
    # `ts` is intentionally NOT set here. `AuditLog.write`'s
    # `setdefault` is the single source of truth for the
    # timestamp — keeps direct-write callers (tests, third-party)
    # consistent with `make_audit_record` callers.
    return record