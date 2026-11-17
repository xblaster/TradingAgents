from datetime import date

import numpy as np
import pandas as pd

from trading_desk.screener import asset_pack, detect_gems, render_screen, screen, stat_score

N = 500
IDX = pd.date_range(end="2026-11-16", periods=N, freq="D")


def frame(close, volume=None):
    close = np.asarray(close, dtype=float)
    vol = np.full(N, 1e6) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": vol}, index=IDX)


def walk(seed, drift=0.0, vol=0.02):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, N)))


def gems_of(f, rank=0.9):
    return {g.rule for g in detect_gems(asset_pack(f), rank)}


def test_emerging_trend_flags_g1():
    rng = np.random.default_rng(1)
    r = np.concatenate([rng.normal(0, 0.02, N - 60), 0.006 + rng.normal(0, 0.01, 60)])
    close = 100 * np.exp(np.cumsum(r))
    close[-1] = close[-2]  # do not end overextended
    assert "G1" in gems_of(frame(close))
    assert "G1" not in gems_of(frame(close), rank=0.3)  # needs relative strength too


def test_volume_breakout_flags_g2():
    close = walk(2, vol=0.01)
    close[-1] = close[-91:-1].max() * 1.03
    volume = np.full(N, 1e6)
    volume[-5:] = 3e6
    assert "G2" in gems_of(frame(close, volume))
    assert "G2" not in gems_of(frame(close))  # same breakout, no volume


def test_capitulation_flags_g3():
    close = walk(3, vol=0.01)
    close[-1] = close[-2] * 0.80
    assert "G3" in gems_of(frame(close))


def test_quiet_random_walk_flags_nothing():
    assert gems_of(frame(walk(4)), rank=0.5) == set()


def test_score_sign():
    up, down = asset_pack(frame(walk(5, drift=0.004, vol=0.01))), asset_pack(frame(walk(5, drift=-0.004, vol=0.01)))
    assert stat_score(up, 0.9) > 0.5 and stat_score(down, 0.1) < -0.5


class Provider:
    def __init__(self, frames):
        self.frames = frames

    async def prices(self, t, start, end):
        return self.frames[t].loc[pd.Timestamp(start) : pd.Timestamp(end)]


async def test_screen_with_cointegration_gem():
    rng = np.random.default_rng(6)
    common = np.cumsum(rng.normal(0, 0.02, N))
    spread = np.zeros(N)
    for i in range(1, N):
        spread[i] = 0.8 * spread[i - 1] + rng.normal(0, 0.01)
    spread[-1] = -0.12  # big dislocation, mean-reverting spread
    frames = {
        "AAA-EUR": frame(100 * np.exp(common + spread)),
        "BBB-EUR": frame(50 * np.exp(common)),
        "CCC-EUR": frame(walk(7)),
    }
    s = await screen(Provider(frames), tuple(frames), as_of=date(2026, 11, 17))
    assert any(g.rule == "G4" for g in s.assets["AAA-EUR"].gems)
    assert s.universe["pairs_tested"] == 3 and s.tests_run > 0
    text = render_screen(s)
    assert "AAA-EUR" in text and "G4 Cointegration dislocation" in text
