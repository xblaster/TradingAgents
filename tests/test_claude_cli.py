from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from trading_desk.llm import CachedBackend, ClaudeCLIBackend, LLMError, LLMRequest, LLMTimeout, extract_json_object

FAKE_CLI = r'''#!{python}
import json, sys, time, os
args = sys.argv[1:]
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_MODE", "ok")
log = os.environ.get("FAKE_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps({{"args": args, "prompt": prompt}}) + "\n")
if mode == "sleep":
    time.sleep(5)
if mode == "garbage":
    print("segfault-ish nonsense"); sys.exit(3)
if mode == "auth":
    print(json.dumps({{"type": "result", "is_error": True, "subtype": "error", "result": "Not logged in. Please run /login"}})); sys.exit(1)
if mode == "flaky":
    marker = os.environ["FAKE_MARKER"]
    if not os.path.exists(marker):
        open(marker, "w").close()
        print(json.dumps({{"type": "result", "is_error": True, "subtype": "error_during_execution", "result": "overloaded"}})); sys.exit(1)
structured = {{"action": "HOLD"}} if "--json-schema" in args else None
result = "Sure! ```json\n{{\"action\": \"SELL\"}}\n```" if mode == "text_only" else "done"
if mode == "text_only":
    structured = None
print("warning: something noisy")
print(json.dumps({{"type": "result", "is_error": False, "subtype": "success", "result": result,
                  "structured_output": structured, "total_cost_usd": 0.0123, "duration_ms": 42,
                  "modelUsage": {{"claude-test-1": {{}}}}}}))
'''

SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "claude"
    path.write_text(FAKE_CLI.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def req(**kw) -> LLMRequest:
    return LLMRequest(system="sys", prompt="the prompt", model="sonnet", schema=SCHEMA, label="t", **kw)


def test_build_args_is_non_interactive_and_isolated():
    args = ClaudeCLIBackend("claude", fallback_model="haiku").build_args(req())
    assert args[:2] == ["claude", "-p"]
    assert args[args.index("--output-format") + 1] == "json"
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--tools") + 1] == ""  # no tools by default
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in args and "--no-session-persistence" in args
    assert json.loads(args[args.index("--json-schema") + 1]) == SCHEMA
    assert args[args.index("--fallback-model") + 1] == "haiku"
    assert "the prompt" not in args  # prompt goes through stdin


def test_build_args_with_tools():
    args = ClaudeCLIBackend().build_args(req(tools=("WebSearch",)))
    assert args[args.index("--tools") + 1] == "WebSearch"
    assert args[args.index("--allowedTools") + 1] == "WebSearch"


async def test_success_reads_structured_output_and_prompt_via_stdin(fake_cli, tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    resp = await ClaudeCLIBackend(str(fake_cli)).complete(req())
    assert resp.data == {"action": "HOLD"}
    assert resp.cost_usd == pytest.approx(0.0123)
    assert resp.model == "claude-test-1"
    assert json.loads(log.read_text())["prompt"] == "the prompt"


async def test_falls_back_to_parsing_json_from_text(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "text_only")
    resp = await ClaudeCLIBackend(str(fake_cli)).complete(req())
    assert resp.data == {"action": "SELL"}


async def test_timeout_kills_the_process(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "sleep")
    with pytest.raises(LLMTimeout):
        await ClaudeCLIBackend(str(fake_cli), timeout_s=0.5, max_retries=0).complete(req())


async def test_auth_error_is_not_retried(fake_cli, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_MODE", "auth")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    with pytest.raises(LLMError) as exc:
        await ClaudeCLIBackend(str(fake_cli), max_retries=3, retry_base_delay_s=0).complete(req())
    assert not exc.value.retryable
    assert len((tmp_path / "log").read_text().splitlines()) == 1


async def test_transient_error_is_retried(fake_cli, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_MODE", "flaky")
    monkeypatch.setenv("FAKE_MARKER", str(tmp_path / "marker"))
    resp = await ClaudeCLIBackend(str(fake_cli), max_retries=2, retry_base_delay_s=0).complete(req())
    assert resp.data == {"action": "HOLD"}


async def test_garbage_output_raises(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "garbage")
    with pytest.raises(LLMError, match="no JSON envelope"):
        await ClaudeCLIBackend(str(fake_cli), max_retries=0).complete(req())


async def test_missing_binary_is_permanent():
    with pytest.raises(LLMError) as exc:
        await ClaudeCLIBackend("/nonexistent/claude").complete(req())
    assert not exc.value.retryable


async def test_cache_hits_skip_the_cli(fake_cli, tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    backend = CachedBackend(ClaudeCLIBackend(str(fake_cli)), tmp_path / "cache")
    first = await backend.complete(req())
    second = await backend.complete(req())
    assert not first.cached and second.cached
    assert second.data == first.data
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('Here you go:\n```json\n{"a": {"b": [1, 2]}}\n```', {"a": {"b": [1, 2]}}),
        ('prefix {"a": "}"} suffix', {"a": "}"}),
        ("no json here", None),
        ("[1, 2, 3]", None),
    ],
)
def test_extract_json_object(text, expected):
    assert extract_json_object(text) == expected
