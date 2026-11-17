from datetime import date

import numpy as np
import pandas as pd

from trading_desk.gem_study import gem_event_study, render_gem_study

N = 700
IDX = pd.date_range("2024-01-01", periods=N, freq="D")


def frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=IDX)


class Provider:
    def __init__(self, frames):
        self.frames = frames

    async def prices(self, t, start, end):
        return self.frames[t].loc[pd.Timestamp(start) : pd.Timestamp(end)]


async def test_event_study_measures_forward_excess_returns():
    rng = np.random.default_rng(0)
    crash_day = 600
    r = rng.normal(0, 0.01, N)
    r[crash_day] = -0.25  # capitulation -> G3 on that day
    r[crash_day + 1 : crash_day + 8] = 0.03  # then a rebound
    frames = {
        "AAA-EUR": frame(100 * np.exp(np.cumsum(r))),
        "BBB-EUR": frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))),
        "CCC-EUR": frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))),
    }
    day = IDX[crash_day].date()
    res = await gem_event_study(Provider(frames), tuple(frames), day, day, step=7, horizons=(7, 30))
    g3 = [e for e in res["events"] if e["rule"] == "G3" and e["ticker"] == "AAA-EUR"]
    assert len(g3) == 1
    assert g3[0]["excess_7d"] > 0.15  # the rebound, measured strictly after the flag
    text = render_gem_study(res)
    assert "G3 Statistical capitulation" in text and "All gems" in text
