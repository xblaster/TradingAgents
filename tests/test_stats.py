import math

import numpy as np
import pandas as pd
import pytest

from trading_desk.stats import cross_asset, diagnostics as dg, performance as pf, regimes, significance as sg

RNG = np.random.default_rng(42)


def random_walk(n=2000, drift=0.0, vol=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, vol, n))))


def ar1_returns(n=3000, phi=0.3, seed=2):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


# ------------------------------------------------------------------ performance

def test_basic_performance_values():
    r = pd.Series([0.1, -0.05, 0.02, -0.01])
    assert pf.max_drawdown(r) == pytest.approx(-0.05)
    assert pf.hit_rate(r) == pytest.approx(0.5)
    assert pf.profit_factor(r) == pytest.approx(0.12 / 0.06)
    assert pf.omega(r) == pytest.approx(0.12 / 0.06)
    assert pf.max_drawdown_duration(pd.Series([0.1, -0.1, 0.01, 0.01, 0.2])) == 3


def test_cagr_and_sharpe_constant_returns():
    r = pd.Series([0.001] * 365)
    assert pf.cagr(r, 365) == pytest.approx(1.001**365 - 1)
    assert pf.sharpe(r, 365) is None  # zero variance


def test_var_cvar_and_cornish_fisher():
    r = pd.Series(RNG.normal(0, 0.01, 200_000))
    assert pf.var_historical(r, 0.05) == pytest.approx(-0.01645, abs=3e-4)
    assert pf.cvar_historical(r, 0.05) == pytest.approx(-0.02063, abs=3e-4)
    assert pf.var_cornish_fisher(r, 0.05) == pytest.approx(-0.01645, abs=3e-4)  # normal: no correction
    skewed = pd.Series(np.concatenate([RNG.normal(0.001, 0.01, 10_000), [-0.15] * 50]))
    assert pf.var_cornish_fisher(skewed) < pf.var_historical(skewed)  # left tail fattened


def test_summary_has_everything():
    s = pf.summary(pd.Series(RNG.normal(0.001, 0.02, 500)), 365)
    for key in ("sortino", "calmar", "omega", "ulcer_index", "cvar_95", "tail_ratio", "excess_kurtosis"):
        assert key in s


# ------------------------------------------------------------------ significance

def test_psr_strong_edge_and_null_distribution():
    good = pd.Series(np.random.default_rng(10).normal(0.002, 0.01, 1000))
    assert sg.probabilistic_sharpe_ratio(good) > 0.99
    # Under the null (no edge) PSR is ~uniform on [0, 1]: check over many independent samples.
    psrs = [sg.probabilistic_sharpe_ratio(pd.Series(np.random.default_rng(s).normal(0, 0.01, 500))) for s in range(200)]
    assert np.mean(psrs) == pytest.approx(0.5, abs=0.06)
    assert np.mean(np.array(psrs) > 0.95) == pytest.approx(0.05, abs=0.04)  # false positives at the 5% level


def test_psr_matches_closed_form_for_normal_returns():
    r = pd.Series(RNG.normal(0.001, 0.01, 2000))
    sr = r.mean() / r.std()
    expected = __import__("scipy").stats.norm.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 + sr**2 / 2))
    assert sg.probabilistic_sharpe_ratio(r) == pytest.approx(expected, abs=0.01)


def test_expected_max_sharpe_grows_with_trials():
    assert sg.expected_max_sharpe(1, 0.01) == 0
    assert sg.expected_max_sharpe(10, 0.01) < sg.expected_max_sharpe(100, 0.01) < sg.expected_max_sharpe(1000, 0.01)
    # Bailey & Lopez de Prado: with V=1, N=1000 trials -> about 3.26
    assert sg.expected_max_sharpe(1000, 1.0) == pytest.approx(3.26, abs=0.02)


def test_deflated_sharpe_penalizes_many_trials():
    r = pd.Series(RNG.normal(0.0012, 0.02, 1000))
    one, many = sg.deflated_sharpe_ratio(r, 1)["dsr"], sg.deflated_sharpe_ratio(r, 200)["dsr"]
    assert many < one


def test_min_track_record_length():
    r = pd.Series(RNG.normal(0.002, 0.01, 1000))
    assert 10 < sg.min_track_record_length(r) < 300
    assert sg.min_track_record_length(-r) == math.inf


def test_bootstrap_ci_covers_and_keeps_length():
    r = pd.Series(RNG.normal(0.001, 0.01, 1000))
    ci = sg.bootstrap_ci(r, sg.annualized_sharpe_stat(365), n_boot=300)
    assert ci["low"] < ci["estimate"] < ci["high"]
    idx = sg.stationary_bootstrap_indices(500, 20, np.random.default_rng(0))
    assert len(idx) == 500 and idx.min() >= 0 and idx.max() < 500


def test_sharpe_difference_detects_better_series():
    base = RNG.normal(0, 0.01, 1500)
    better = base + 0.001
    out = sg.sharpe_difference_test(pd.Series(better), pd.Series(base), n_boot=300)
    assert out["difference"] > 0 and out["p_a_better"] > 0.95


def test_mean_tstat():
    assert sg.mean_return_tstat(pd.Series(RNG.normal(0.01, 0.01, 200)))["p_value_positive_mean"] < 0.001


# ------------------------------------------------------------------ diagnostics

def test_hurst_and_variance_ratio_on_random_walk():
    p = random_walk()
    assert dg.hurst_exponent(p) == pytest.approx(0.5, abs=0.08)
    vr = dg.variance_ratio(p, 5)
    assert vr["vr"] == pytest.approx(1.0, abs=0.1) and vr["p_value"] > 0.01


def test_variance_ratio_detects_momentum_and_reversal():
    mom = pd.Series(100 * np.exp(np.cumsum(ar1_returns(phi=0.3))))
    rev = pd.Series(100 * np.exp(np.cumsum(ar1_returns(phi=-0.3))))
    assert dg.variance_ratio(mom, 5)["vr"] > 1.2 and dg.variance_ratio(mom, 5)["p_value"] < 0.01
    assert dg.variance_ratio(rev, 5)["vr"] < 0.8 and dg.variance_ratio(rev, 5)["p_value"] < 0.01
    assert dg.diagnose(mom)["character"] == "trending (momentum)"
    assert dg.diagnose(rev)["character"] == "mean-reverting"


def test_ljung_box_and_acf():
    assert dg.ljung_box(pd.Series(ar1_returns(phi=0.3)))["p_value"] < 0.001
    assert dg.ljung_box(pd.Series(RNG.normal(0, 1, 2000)))["p_value"] > 0.001
    assert dg.autocorrelations(pd.Series(ar1_returns(phi=0.3)))["acf_1"] == pytest.approx(0.3, abs=0.05)


def test_stationarity_tests():
    noise = pd.Series(RNG.normal(0, 1, 1000))
    walk = noise.cumsum()
    assert dg.adf(noise)["p_value"] < 0.01 and dg.adf(walk)["p_value"] > 0.05
    assert dg.kpss(noise)["p_value"] > 0.05


def test_trend_tstat_and_half_life():
    up = random_walk(400, drift=0.004, vol=0.01, seed=3)
    t = dg.trend_tstat(up, 120)
    assert t["t_stat"] > 3 and t["slope_annual"] > 0.5
    rng = np.random.default_rng(5)
    x = np.zeros(3000)
    for i in range(1, 3000):
        x[i] = 0.9 * x[i - 1] + rng.normal()  # half-life ln2/ln(1/0.9) = 6.58
    assert dg.half_life(pd.Series(x)) == pytest.approx(6.58, rel=0.15)
    assert dg.half_life(pd.Series(np.cumsum(rng.normal(size=3000)) + np.arange(3000))) is None  # trend, no reversion


def test_variance_ratio_z_is_standard_normal_under_null():
    zs = [dg.variance_ratio(random_walk(1000, seed=s), 5)["z"] for s in range(150)]
    assert np.mean(zs) == pytest.approx(0, abs=0.25) and np.std(zs) == pytest.approx(1, abs=0.2)


def test_robust_zscore_and_jb():
    s = pd.Series(np.concatenate([RNG.normal(0, 1, 100), [8.0]]))
    assert dg.robust_zscore(s) > 5
    assert dg.jarque_bera(pd.Series(RNG.standard_t(3, 5000)))["p_value"] < 0.001


# ------------------------------------------------------------------ regimes & cross-asset

def test_volatility_and_trend_regimes():
    calm_then_wild = pd.Series(100 * np.exp(np.cumsum(np.concatenate([RNG.normal(0, 0.01, 400), RNG.normal(0, 0.05, 30)]))))
    assert regimes.volatility_regime(calm_then_wild)["label"] in ("high", "extreme")
    assert regimes.trend_regime(random_walk(400, drift=0.004, vol=0.01, seed=3))["label"] == "uptrend"
    rng = np.random.default_rng(8)
    # long rally, then a 40-day drop: 60d trend significantly down, 120d still significantly up
    rally_then_drop = np.concatenate([0.006 + rng.normal(0, 0.01, 360), -0.010 + rng.normal(0, 0.01, 40)])
    tr = regimes.trend_regime(pd.Series(100 * np.exp(np.cumsum(rally_then_drop))))
    assert tr["t_stat_60d"] < -2 and tr["t_stat_120d"] > 2 and tr["label"] == "transition"


def test_markov_switching_finds_high_vol_state():
    r = np.concatenate([RNG.normal(0, 0.01, 500), RNG.normal(0, 0.05, 150)])
    out = regimes.markov_switching(pd.Series(100 * np.exp(np.cumsum(r))))
    assert out is not None and out["p_high_vol_state"] > 0.8
    assert out["high_state_ann_vol"] > 2 * out["low_state_ann_vol"]


def test_cross_asset_tools():
    n = 500
    common = np.cumsum(RNG.normal(0, 0.02, n))
    a = pd.Series(100 * np.exp(common + RNG.normal(0, 0.01, n)))
    b = pd.Series(50 * np.exp(common + RNG.normal(0, 0.01, n)))  # cointegrated with a
    c = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0.002, 0.02, n))))
    closes = pd.DataFrame({"A": a, "B": b, "C": c})
    assert cross_asset.correlation_matrix(closes).loc["A", "B"] > 0.5
    # daily returns: common shock var 4e-4, each idiosyncratic level noise adds 2e-4 -> beta = 4/6
    assert cross_asset.beta(closes, "A", "B") == pytest.approx(0.667, abs=0.15)
    ranks = cross_asset.momentum_ranks(closes)
    assert set(ranks["rank_avg"].round(3)) <= {round(x, 3) for x in np.linspace(1 / 3, 1, 7)}
    hits = cross_asset.cointegration_scan(closes)
    assert any({h["a"], h["b"]} == {"A", "B"} for h in hits)
