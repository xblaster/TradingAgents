"""Trend-following crypto portfolio for Revolut X (spec: STRATEGY.md, pre-registered).

Signal per asset: share of lookbacks L in {20, 60, 120, 250} days for which close_t > close_{t-L}.
Weights: inverse 60-day volatility across eligible assets, multiplied by the signal. The rest
is held in EUR. Long-only, no leverage.

Everything here is computed from data up to the decision bar only; the simulator fills at the
next open and charges costs on turnover, like `backtest.simulate` does for single assets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data import DataProvider

CORE = ("BTC-EUR", "ETH-EUR")
EXTENDED = CORE + ("SOL-EUR", "XRP-EUR", "ADA-EUR", "DOGE-EUR", "LINK-EUR", "DOT-EUR", "AVAX-EUR", "LTC-EUR")
UNIVERSES = {"core": CORE, "extended": EXTENDED}
DAYS = 365


@dataclass(frozen=True)
class TrendConfig:
    lookbacks: tuple[int, ...] = (20, 60, 120, 250)
    vol_window: int = 60
    min_history: int = 250
    every: int = 7
    min_trade: float = 0.02
    cost_bps: float = 15.0

    def scaled(self, k: float) -> "TrendConfig":
        return replace(self, lookbacks=tuple(max(2, round(l * k)) for l in self.lookbacks))


# ------------------------------------------------------------------- signals


def trend_signals(closes: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Fraction of lookbacks with a positive trailing return; NaN until the longest lookback exists."""
    votes = [(closes > closes.shift(l)).astype(float).where(closes.shift(l).notna()) for l in lookbacks]
    return sum(votes) / len(votes)


def target_weights(closes: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """Target weight of every asset at every close, using data up to that close only."""
    signals = trend_signals(closes, cfg.lookbacks)
    vol = np.log(closes).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window).std() * math.sqrt(DAYS)
    eligible = (closes.notna().cumsum() >= cfg.min_history) & signals.notna() & vol.gt(0) & closes.notna()
    inv_vol = (1 / vol).where(eligible)
    budget = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return (budget * signals).where(eligible, 0.0).fillna(0.0)


def equal_weight_targets(closes: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    eligible = (closes.notna().cumsum() >= cfg.min_history) & closes.notna()
    n = eligible.sum(axis=1).replace(0, np.nan)
    return eligible.astype(float).div(n, axis=0).fillna(0.0)


def single_asset_targets(closes: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    out[ticker] = closes[ticker].notna().astype(float)
    return out


# ----------------------------------------------------------------- simulator


@dataclass
class PortfolioRun:
    dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    exposure: list[float] = field(default_factory=list)
    turnover: float = 0.0
    fees: float = 0.0
    trades: int = 0
    last_weights: dict[str, float] = field(default_factory=dict)


def simulate_portfolio(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    targets: pd.DataFrame,
    start: date,
    end: date,
    *,
    every: int,
    min_trade: float,
    cost_bps: float,
) -> PortfolioRun:
    """Decide on the close of day t (row t of `targets`), fill at the open of t+1."""
    opens, closes = opens.ffill(), closes.ffill()  # a held asset with a missing bar simply does not move
    idx = closes.index
    start_i = int(idx.searchsorted(pd.Timestamp(start)))
    end_i = int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    if end_i - start_i < 1:
        raise ValueError("Backtest window too short")
    o, c, tgt = opens.to_numpy(float), closes.to_numpy(float), targets.reindex(idx).fillna(0.0).to_numpy(float)
    cost = cost_bps / 10_000
    n = closes.shape[1]
    w = np.zeros(n)
    equity = 1.0
    pending: np.ndarray | None = None
    run = PortfolioRun()

    def move(w: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.nan_to_num(r)
        growth = 1 + float(w @ r)
        return (w * (1 + r) / growth if growth > 0 else np.zeros_like(w)), growth

    for i in range(start_i, end_i + 1):
        if i > start_i:
            w, g = move(w, o[i] / c[i - 1] - 1)
            equity *= g
            if pending is not None:
                diff = pending - w
                trade = (np.abs(diff) >= min_trade) | ((pending == 0) & (w > 0))
                turnover = float(np.abs(diff[trade]).sum())
                if turnover > 0:
                    equity *= 1 - turnover * cost
                    run.fees += turnover * cost
                    run.turnover += turnover
                    run.trades += int(trade.sum())
                    w = np.where(trade, pending, w)
                pending = None
            w, g = move(w, c[i] / o[i] - 1)
            equity *= g
        run.dates.append(idx[i].date())
        run.equity.append(equity)
        run.exposure.append(float(w.sum()))
        if (i - start_i) % every == 0 and i < end_i:
            pending = tgt[i]
    run.last_weights = {col: float(x) for col, x in zip(closes.columns, w) if x > 0}
    return run


def portfolio_metrics(run: PortfolioRun) -> dict[str, float | None]:
    eq = pd.Series(run.equity)
    rets = eq.pct_change().dropna()
    years = max(len(rets) / DAYS, 1e-9)
    std = float(rets.std()) if len(rets) > 1 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr,
        "ann_vol": std * math.sqrt(DAYS),
        "sharpe": float(rets.mean() / std * math.sqrt(DAYS)) if std > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else None,
        "avg_exposure": float(np.mean(run.exposure)),
        "turnover_per_year": run.turnover / years,
        "fees": run.fees,
    }


# ------------------------------------------------------------------- data


async def load_prices(provider: DataProvider, tickers: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned daily opens and closes; NaN before an asset's first bar."""
    frames = {t: await provider.prices(t, start, end) for t in tickers}
    closes = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
    opens = pd.DataFrame({t: f["Open"] for t, f in frames.items()}).reindex(closes.index)
    return opens, closes


def warmup_start(start: date, cfg: TrendConfig) -> date:
    return start - timedelta(days=int(max(max(cfg.lookbacks), cfg.min_history, cfg.vol_window) * 1.6) + 30)
