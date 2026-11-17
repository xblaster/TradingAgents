"""Run Claude through the Claude Code CLI (`claude -p`) as an async subprocess.

Each call is a single, isolated, non-interactive turn:
  * `-p` (print mode) never waits for user input,
  * `--output-format json` returns a machine-readable envelope (result, cost, errors),
  * `--json-schema` makes the CLI validate structured output, returned in `structured_output`,
  * `--system-prompt` replaces the coding-agent prompt with ours,
  * `--setting-sources ""`, `--strict-mcp-config` and an empty working directory keep
    the user's settings, hooks, MCP servers and CLAUDE.md files out of the call,
  * `--tools ""` disables every tool unless the request explicitly asks for some.

The prompt goes through stdin, never argv, so its size is not limited by ARG_MAX and it
does not show up in `ps`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from typing import Any, Sequence

from .base import LLMError, LLMRequest, LLMResponse, LLMTimeout

log = logging.getLogger(__name__)

_PERMANENT_HINTS = ("not logged in", "invalid api key", "please run /login", "authentication")


class ClaudeCLIBackend:
    def __init__(
        self,
        command: str = "claude",
        *,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        fallback_model: str | None = None,
        max_concurrency: int = 4,
        extra_args: Sequence[str] = (),
        retry_base_delay_s: float = 2.0,
    ):
        self.command = command
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.fallback_model = fallback_model
        self.extra_args = list(extra_args)
        self.retry_base_delay_s = retry_base_delay_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._workdir = tempfile.mkdtemp(prefix="desk-claude-")

    # ------------------------------------------------------------------ public

    def build_args(self, request: LLMRequest) -> list[str]:
        args = [
            self.command,
            "-p",
            "--output-format", "json",
            "--model", request.model,
            "--system-prompt", request.system,
            "--no-session-persistence",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--tools", ",".join(request.tools),
        ]
        if request.tools:
            args += ["--allowedTools", ",".join(request.tools)]
        if request.schema is not None:
            args += ["--json-schema", json.dumps(request.schema, separators=(",", ":"))]
        if self.fallback_model and self.fallback_model != request.model:
            args += ["--fallback-model", self.fallback_model]
        return args + self.extra_args

    async def complete(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    return await self._run_once(request)
            except LLMError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self.retry_base_delay_s * (2**attempt)
                attempt += 1
                log.warning("%s: %s; retry %d/%d in %.0fs", request.label or "llm", exc, attempt, self.max_retries, delay)
                await asyncio.sleep(delay)

    # ----------------------------------------------------------------- private

    async def _run_once(self, request: LLMRequest) -> LLMResponse:
        if shutil.which(self.command) is None and not os.path.isfile(self.command):
            raise LLMError(f"Claude CLI not found: {self.command!r}. Install Claude Code or set llm.command.", retryable=False)

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *self.build_args(request),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workdir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(request.prompt.encode()), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise LLMTimeout(f"claude -p timed out after {self.timeout_s:.0f}s")
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace").strip()
        envelope = _parse_envelope(out)

        if envelope is None:
            raise LLMError(_describe_failure(proc.returncode, err, out), retryable=_is_retryable(err + out))
        if proc.returncode != 0 or envelope.get("is_error"):
            message = str(envelope.get("result") or envelope.get("subtype") or err or "unknown error")
            raise LLMError(f"claude -p failed ({envelope.get('subtype')}): {message[:500]}", retryable=_is_retryable(message))

        text = str(envelope.get("result") or "")
        data: dict[str, Any] | None = None
        if request.schema is not None:
            data = envelope.get("structured_output")
            if not isinstance(data, dict):
                data = extract_json_object(text)
            if data is None:
                raise LLMError("Structured output requested but the response contains no JSON object")

        models = list((envelope.get("modelUsage") or {}).keys())
        return LLMResponse(
            text=text,
            data=data,
            model=models[0] if models else request.model,
            cost_usd=float(envelope.get("total_cost_usd") or 0.0),
            duration_ms=int(envelope.get("duration_ms") or (time.monotonic() - started) * 1000),
            meta={"num_turns": envelope.get("num_turns"), "session_id": envelope.get("session_id")},
        )


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        # Some versions print warnings before the JSON envelope: keep the last line that parses.
        for line in reversed(stdout.splitlines()):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            return None
    if isinstance(payload, list):  # stream-ish outputs: the result event is last
        payload = next((e for e in reversed(payload) if isinstance(e, dict) and e.get("type") == "result"), None)
    return payload if isinstance(payload, dict) else None


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from free text (fenced or inline)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for start in (i for i, ch in enumerate(candidate) if ch == "{"):
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _is_retryable(message: str) -> bool:
    lowered = message.lower()
    return not any(hint in lowered for hint in _PERMANENT_HINTS)


def _describe_failure(returncode: int | None, stderr: str, stdout: str) -> str:
    detail = stderr or stdout.strip() or "no output"
    return f"claude -p exited with code {returncode} and no JSON envelope: {detail[:500]}"
