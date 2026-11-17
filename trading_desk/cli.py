"""Command line: `desk analyze`, `desk backtest`, `desk doctor`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from datetime import date

from .config import DeskSettings
from .data import YahooProvider
from .llm import CachedBackend, ClaudeCLIBackend, LLMBackend, LLMRequest


def build_backend(settings: DeskSettings, use_cache: bool = True) -> LLMBackend:
    llm = settings.llm
    backend: LLMBackend = ClaudeCLIBackend(
        llm.command, timeout_s=llm.timeout_s, max_retries=llm.max_retries,
        fallback_model=llm.fallback_model, max_concurrency=llm.max_concurrency,
    )
    if use_cache and llm.cache_dir:
        backend = CachedBackend(backend, llm.cache_dir)
    return backend


def _settings(args: argparse.Namespace) -> DeskSettings:
    s = DeskSettings.load(args.config)
    if args.fast_model:
        s.llm.fast_model = args.fast_model
    if args.deep_model:
        s.llm.deep_model = args.deep_model
    if args.lang:
        s.language = args.lang
    if args.rounds:
        s.debate_rounds = args.rounds
    if args.no_web:
        s.web_search = False
    return s


async def _analyze(args: argparse.Namespace) -> int:
    from .pipeline import TradingDesk

    settings = _settings(args)
    desk = TradingDesk(settings, build_backend(settings, not args.no_cache), YahooProvider())
    run = await desk.run(args.ticker, args.date, current_position_pct=args.position)
    folder = run.save(settings.results_dir)
    if args.json:
        print(run.decision.model_dump_json(indent=2))
        return 0
    d = run.decision
    print(f"\n{run.ticker} @ {run.snapshot.last_close} ({run.as_of}, {run.snapshot.mode})")
    print(f"  Decision : {d.action} -> {d.target_position_pct:.1f}% (from {d.current_position_pct:.1f}%), conviction {d.conviction:.2f}")
    print(f"  Stop     : {d.stop_loss or '-'}   Take-profit: {d.take_profit or '-'}   Horizon: {d.horizon_days}d")
    for name, rep in run.analysts.items():
        print(f"  {name:<12}: {rep.stance} ({rep.confidence:.2f})")
    if run.risk:
        print(f"  risk        : {run.risk.risk_level}, max {run.risk.max_position_pct:.0f}%{' VETO' if run.risk.veto else ''}")
    for note in d.adjustments:
        print(f"  ! {note}")
    print(f"  Cost ${run.cost_usd:.4f}, report: {folder / 'report.md'}")
    return 0


async def _backtest(args: argparse.Namespace) -> int:
    from .backtest import BASELINES, run_backtest
    from .pipeline import TradingDesk
    from .report import strategy_table

    settings = _settings(args)
    if args.max_position is not None:
        settings.risk.max_position_pct = args.max_position
    if args.knowledge_cutoff:
        settings.llm.knowledge_cutoff = args.knowledge_cutoff
    strategies = tuple(BASELINES) if args.no_llm else tuple(s.strip() for s in args.strategies.split(","))
    provider = YahooProvider()
    desk = TradingDesk(settings, build_backend(settings, not args.no_cache), provider) if "llm" in strategies else None

    def on_decision(run) -> None:
        run.save(settings.results_dir)  # every decision stays auditable
        d = run.decision
        print(f"  {run.as_of}  {d.action:<4} -> {d.target_position_pct:5.1f}%  conv {d.conviction:.2f}  (${run.cost_usd:.3f})", flush=True)

    result = await run_backtest(
        desk, provider, args.ticker, args.start, args.end, every=args.every, cost_bps=args.cost_bps,
        max_position_pct=settings.risk.max_position_pct, strategies=strategies, on_decision=on_decision,
    )
    folder = result.save(settings.results_dir)
    print(f"\n{result.ticker} {result.start} -> {result.end}, every {result.every} bars, {result.cost_bps:g} bps, cap {result.max_position_pct:g}%\n")
    for w in result.warnings:
        print(f"! {w}")
    print("\n".join(strategy_table(result)))
    print(f"\nReport: {folder / 'report.md'}")
    return 0


async def _doctor(args: argparse.Namespace) -> int:
    settings = DeskSettings.load(args.config)
    ok = True
    path = shutil.which(settings.llm.command)
    print(f"[{'ok' if path else 'FAIL'}] Claude CLI: {path or 'not found'}")
    if path:
        backend = ClaudeCLIBackend(settings.llm.command, timeout_s=120, max_retries=0)
        try:
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
            resp = await backend.complete(LLMRequest(system="Reply with JSON.", prompt="Return ok=true.", model=settings.llm.fast_model, schema=schema))
            print(f"[ok] claude -p structured call: {resp.data} via {resp.model} in {resp.duration_ms} ms (${resp.cost_usd:.4f})")
        except Exception as exc:
            ok = False
            print(f"[FAIL] claude -p: {exc}")
    try:
        snap = await YahooProvider().snapshot("SPY", date.today(), 400)
        print(f"[ok] Yahoo data: SPY close {snap.last_close} on {snap.last_bar_date}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] Yahoo data: {exc}")

    from .revolutx import RevolutXClient, summarize_order_book

    client = RevolutXClient.from_env()
    try:
        book = summarize_order_book(await client.public_order_book("BTC-EUR"))
        print(f"[ok] Revolut X public order book: BTC-EUR bid {book['best_bid']} / ask {book['best_ask']} ({book['spread_bps']} bps)")
    except Exception as exc:
        ok = False
        print(f"[FAIL] Revolut X: {exc}")
    if client.authenticated:
        try:
            balances = await client.balances()
            print(f"[ok] Revolut X authenticated: {len(balances)} balances visible")
        except Exception as exc:
            ok = False
            print(f"[FAIL] Revolut X authentication: {exc}")
    else:
        print("[--] Revolut X API key not configured (REVOLUTX_API_KEY): public data only")
    return 0 if ok else 1


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="desk", description="Multi-agent trading desk powered by the Claude Code CLI.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="TOML settings file")
    common.add_argument("--fast-model", help="Model for analysts and debaters (default: sonnet)")
    common.add_argument("--deep-model", help="Model for risk and portfolio managers (default: opus)")
    common.add_argument("--lang", help="Language of the reports, e.g. French")
    common.add_argument("--rounds", type=int, help="Bull/bear debate rounds")
    common.add_argument("--no-web", action="store_true", help="Do not let the news analyst search the web")
    common.add_argument("--no-cache", action="store_true", help="Bypass the LLM response cache")

    a = sub.add_parser("analyze", parents=[common], help="Run the desk on one ticker")
    a.add_argument("ticker")
    a.add_argument("--date", type=date.fromisoformat, default=date.today(), help="Analysis date (default: today)")
    a.add_argument("--position", type=float, default=0.0, help="Current position, %% of portfolio")
    a.add_argument("--json", action="store_true", help="Print the final decision as JSON")

    b = sub.add_parser("backtest", parents=[common], help="Walk-forward backtest on one ticker")
    b.add_argument("ticker")
    b.add_argument("--start", type=date.fromisoformat, required=True)
    b.add_argument("--end", type=date.fromisoformat, default=date.today())
    b.add_argument("--every", type=int, default=5, help="Bars between decisions (crypto: days)")
    b.add_argument("--cost-bps", type=float, help="Cost per unit of turnover in bps (default: 15 crypto, 5 equity)")
    b.add_argument("--max-position", type=float, help="Position cap in %% for every strategy (default: risk.max_position_pct)")
    b.add_argument("--strategies", default="llm,buy_hold,trend_sma,momentum", help="Comma-separated: llm,buy_hold,trend_sma,momentum")
    b.add_argument("--no-llm", action="store_true", help="Baselines only: free, fast, any period length")
    b.add_argument("--knowledge-cutoff", type=date.fromisoformat, help="Models' training cutoff, to flag contaminated periods")

    d = sub.add_parser("doctor", help="Check the Claude CLI and the data source")
    d.add_argument("--config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    handler = {"analyze": _analyze, "backtest": _backtest, "doctor": _doctor}[args.command]
    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
