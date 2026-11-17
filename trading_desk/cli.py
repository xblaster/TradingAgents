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


async def _evaluate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .evaluation import evaluate, render

    ev = await evaluate(YahooProvider(), end=args.end)
    text = render(ev)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    for c in ev.criteria:
        print(f"  {c.code} {'PASS' if c.passed else 'FAIL'}  {c.label}\n        {c.detail}")
    print(f"\nVerdict: {'VIABLE' if ev.viable else 'NOT VIABLE'}   (report: {out})")
    return 0 if ev.viable else 2


async def _paper(args: argparse.Namespace) -> int:
    from .paper import STRATEGIES, PaperTrader, revolutx_quote

    trader = PaperTrader(args.folder, YahooProvider(), revolutx_quote)
    if args.action == "run":
        entry = await trader.run(force=args.force)
        for name, b in entry["books"].items():
            orders = ", ".join(f"{o['side']} {o['pair']} {o['eur']:.0f}EUR" for o in b["orders"]) or "-"
            print(f"  {name:<7} NAV {b['nav']:>10.2f}  {'rebalanced' if b['rebalanced'] else 'held      '}  {orders}")
        print(f"\nJournal: {trader.journal_path}")
        return 0
    st = trader.status()
    if not st["books"]:
        print("No paper trading yet: run `desk paper run` once a day.")
        return 0
    print(f"Forward window {st['start']} -> {st['last']} ({st['weeks']} weeks)\n")
    print(f"  {'book':<7} {'NAV':>10} {'return':>8} {'max DD':>8} {'Sharpe':>7}  verdict")
    for name, s in st["books"].items():
        sharpe = f"{s['sharpe']:.2f}" if s["sharpe"] is not None else "n/a"
        verdict = s.get("verdict", "benchmark")
        print(f"  {name:<7} {s['nav']:>10.2f} {s['return'] * 100:>+7.1f}% {s['max_drawdown'] * 100:>+7.1f}% {sharpe:>7}  {verdict}")
    return 0


async def _portfolio(args: argparse.Namespace) -> int:
    from .paper import STRATEGIES, Book, PaperTrader, plan_orders, revolutx_quote
    from .portfolio import EXTENDED

    if args.from_revolutx:
        from .revolutx import RevolutXClient

        client = RevolutXClient.from_env()
        balances = await client.balances()
        cash = sum(float(b["total"]) for b in balances if b["currency"] == "EUR")
        holdings = {f"{b['currency']}-EUR": float(b["total"]) for b in balances if f"{b['currency']}-EUR" in EXTENDED and float(b["total"]) > 0}
    else:
        cash = args.cash
        holdings = {k.strip().upper(): float(v) for k, v in (kv.split("=") for kv in args.holdings.split(",") if kv)} if args.holdings else {}

    trader = PaperTrader(".", YahooProvider(), revolutx_quote)
    targets = (await trader.targets(date.today()))[args.strategy]
    quotes = {p: await revolutx_quote(p) for p in EXTENDED}
    book = Book(cash, holdings)
    nav, current = book.nav(quotes), book.weights(quotes)
    print(f"Strategy {args.strategy} (lookbacks {STRATEGIES[args.strategy].lookbacks}), NAV {nav:.2f} EUR, signals from completed bars up to {date.today()}\n")
    print(f"  {'pair':<9} {'current':>8} {'target':>8}")
    for pair in sorted(set(targets) | set(current), key=lambda p: -targets.get(p, 0)):
        print(f"  {pair:<9} {current.get(pair, 0) * 100:>7.1f}% {targets.get(pair, 0) * 100:>7.1f}%")
    print(f"  {'EUR':<9} {cash / nav * 100 if nav else 0:>7.1f}% {(1 - sum(targets.values())) * 100:>7.1f}%")
    orders = plan_orders(book, targets, quotes)
    if not orders:
        print("\nNo order needed (every gap is below 2% of the portfolio).")
        return 0
    print("\nOrders to place on Revolut X (limit post-only at the touch = 0% maker fee; market = 0.09% taker):")
    for o in orders:
        q = quotes[o.pair]
        limit = q.bid if o.side == "buy" else q.ask
        print(f"  {o.side.upper():<4} {o.pair:<9} {o.qty:.8f} (~{o.eur:.2f} EUR)   limit {limit}   market ~{o.price}")
    print("\nNothing was sent: this command never places orders.")
    return 0


def _universe(spec: str) -> tuple[str, ...]:
    from .portfolio import UNIVERSES

    return UNIVERSES.get(spec) or tuple(t.strip().upper() for t in spec.split(",") if t.strip())


async def _screen(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .decision import HybridDesk
    from .pipeline import TradingDesk
    from .screener import RULES, render_screen, screen

    provider = YahooProvider()
    sc = await screen(provider, _universe(args.universe), args.date)
    text = render_screen(sc)
    out = Path(args.output or f"results/screens/{sc.as_of}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)

    gem_assets = list(dict.fromkeys(t for t, _ in sc.gems))[: args.max_llm]
    if not args.llm or not gem_assets:
        print(f"LLM calls: 0 ({'no gem' if not gem_assets else 'use --llm to review gems'}). Report: {out}")
        return 0
    settings = _settings(args)
    hybrid = HybridDesk(settings, provider, TradingDesk(settings, build_backend(settings, not args.no_cache), provider),
                        universe=_universe(args.universe), llm_last_resort=False)
    total = 0.0
    print(f"LLM review of {len(gem_assets)} gem(s):")
    for t in gem_assets:
        d = await hybrid.decide(t, args.date)
        total += d.llm_cost_usd
        rules = ", ".join(f"{g['rule']} {RULES[g['rule']].name}" for g in d.gems)
        print(f"  {t:<9} [{rules}] -> {d.action} {d.target_position_pct:.1f}% ({d.decided_by}, ${d.llm_cost_usd:.3f})")
        if d.llm_run:
            d.llm_run.save(settings.results_dir)
    print(f"LLM cost ${total:.3f}. Report: {out}")
    return 0


async def _decide(args: argparse.Namespace) -> int:
    from .decision import HybridDesk
    from .pipeline import TradingDesk

    settings = _settings(args)
    provider = YahooProvider()
    llm = None if args.no_llm else TradingDesk(settings, build_backend(settings, not args.no_cache), provider)
    universe = _universe(args.universe) if args.universe else None
    hybrid = HybridDesk(settings, provider, llm, universe=universe, llm_last_resort=not args.no_last_resort)
    d = await hybrid.decide(args.ticker, args.date, current_position_pct=args.position)
    print(f"\n{d.ticker} {d.as_of}: {d.action} -> {d.target_position_pct:.1f}% (from {d.current_position_pct:.1f}%), stop {d.stop_loss or '-'}")
    print(f"  decided by: {d.decided_by}   LLM cost: ${d.llm_cost_usd:.3f}")
    print(f"  statistics: {d.stats.action} {d.stats.target_position_pct:.1f}%, {'clear' if d.stats.clear else 'ambiguous'}; " + "; ".join(d.stats.reasons))
    for g in d.gems:
        print(f"  gem {g['rule']}: {g['evidence']}")
    if d.llm_run:
        folder = d.llm_run.save(settings.results_dir)
        print(f"  LLM report: {folder / 'report.md'}")
    return 0


async def _stats(args: argparse.Namespace) -> int:
    from datetime import timedelta

    import numpy as np

    from .data.models import asset_class_of, periods_per_year
    from .stats import diagnostics as dg, performance as pf, regimes, significance as sg

    ticker = args.ticker.upper()
    ppy = periods_per_year(asset_class_of(ticker))
    bars = await YahooProvider().prices(ticker, args.date - timedelta(days=args.days), args.date)
    close = bars["Close"].dropna()
    r = close.pct_change().dropna()

    def show(title: str, d: dict) -> None:
        print(f"\n{title}")
        for k, v in d.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items())
            elif isinstance(v, float):
                v = f"{v:.4g}"
            print(f"  {k:<28} {v}")

    print(f"{ticker}: {len(close)} daily bars, {close.index[0].date()} -> {close.index[-1].date()}, {ppy} periods/year")
    show("Performance and risk", pf.summary(r, ppy))
    show("Significance", {
        "psr_sharpe_above_0": sg.probabilistic_sharpe_ratio(r),
        "min_track_record_days_95": sg.min_track_record_length(r),
        **{f"mean_{k}": v for k, v in sg.mean_return_tstat(r).items()},
        "sharpe_bootstrap_90": sg.bootstrap_ci(r, sg.annualized_sharpe_stat(ppy), n_boot=500),
        "cagr_bootstrap_90": sg.bootstrap_ci(r, sg.cagr_stat(ppy), n_boot=500),
    })
    diag = dg.diagnose(close, ppy)
    show("Time-series diagnostics", {k: v for k, v in diag.items() if v is not None})
    show("Regimes", {"volatility": regimes.volatility_regime(close, ppy=ppy), "trend": regimes.trend_regime(close, ppy),
                     "markov_switching": regimes.markov_switching(close) or "did not converge"})
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

    e = sub.add_parser("evaluate", help="Run the pre-registered evaluation of the Revolut X trend strategy (STRATEGY.md)")
    e.add_argument("--end", type=date.fromisoformat, help="End of the out-of-sample window (default: protocol date)")
    e.add_argument("--output", default="reports/strategy_evaluation.md")

    pp = sub.add_parser("paper", help="Forward paper trading of v1/v2 on live Revolut X prices (run once a day)")
    pp.add_argument("action", choices=["run", "status"])
    pp.add_argument("--folder", default="paper")
    pp.add_argument("--force", action="store_true", help="Run again today")

    pf = sub.add_parser("portfolio", help="Today's target weights and the Revolut X orders to reach them (never sends orders)")
    pf.add_argument("--strategy", choices=["v1", "v2"], default="v1")
    pf.add_argument("--cash", type=float, default=1000.0, help="EUR cash")
    pf.add_argument("--holdings", help="e.g. BTC-EUR=0.01,ETH-EUR=0.5")
    pf.add_argument("--from-revolutx", action="store_true", help="Read balances with REVOLUTX_API_KEY")

    sc = sub.add_parser("screen", parents=[common], help="Statistical screen of a universe; the LLM only reviews gems (--llm)")
    sc.add_argument("--universe", default="extended", help="core | extended | comma-separated tickers")
    sc.add_argument("--date", type=date.fromisoformat, default=date.today())
    sc.add_argument("--llm", action="store_true", help="Send flagged gems to the LLM desk")
    sc.add_argument("--max-llm", type=int, default=3, help="At most this many gems reviewed by the LLM")
    sc.add_argument("--output", help="Markdown report path")

    dc = sub.add_parser("decide", parents=[common], help="Stats-first decision; the LLM only for gems or as last resort")
    dc.add_argument("ticker")
    dc.add_argument("--date", type=date.fromisoformat, default=date.today())
    dc.add_argument("--position", type=float, default=0.0)
    dc.add_argument("--universe", help="Universe for ranks/cointegration (default: Revolut X pairs for crypto)")
    dc.add_argument("--no-llm", action="store_true", help="Statistics only")
    dc.add_argument("--no-last-resort", action="store_true", help="Hold instead of asking the LLM when statistics are ambiguous")

    stt = sub.add_parser("stats", help="Full statistical report on one asset (no LLM)")
    stt.add_argument("ticker")
    stt.add_argument("--date", type=date.fromisoformat, default=date.today())
    stt.add_argument("--days", type=int, default=730, help="History length")

    d = sub.add_parser("doctor", help="Check the Claude CLI and the data source")
    d.add_argument("--config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    handler = {"analyze": _analyze, "backtest": _backtest, "evaluate": _evaluate, "paper": _paper, "portfolio": _portfolio,
               "screen": _screen, "decide": _decide, "stats": _stats, "doctor": _doctor}[args.command]
    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
