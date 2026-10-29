"""Kernels, input geometry and the sparse (VFE) GP against the exact GP."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from clara_gapfill import gp

P = {"sigma_A": torch.tensor(1.3, dtype=gp.DTYPE), "sigma_B": torch.tensor(0.7, dtype=gp.DTYPE),
     "ell_space_A": torch.tensor(0.3, dtype=gp.DTYPE), "ell_season": torch.tensor(1.0, dtype=gp.DTYPE),
     "ell_space_B": torch.tensor(0.6, dtype=gp.DTYPE), "ell_time": torch.tensor(1.5, dtype=gp.DTYPE),
     "sigma0": torch.tensor(0.4, dtype=gp.DTYPE), "sigma1": torch.tensor(0.5, dtype=gp.DTYPE),
     "beta0": torch.tensor(0.1, dtype=gp.DTYPE), "beta1": torch.tensor(0.6, dtype=gp.DTYPE)}


def random_inputs(n, seed=0, with_special=True):
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    if with_special:  # poles and both sides of the dateline
        lat[:4], lon[:4] = [90, -90, 10, 10], [0, 45, 179.9, -179.9]
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.uniform(0, 6.7 * 365, n), unit="D")
    return gp.embed(lat, lon, time, "2020-01-01")


def test_embed_geometry():
    x = gp.embed([90, 90, 10, 10], [0, 123, 179.9, -179.9], pd.to_datetime(["2021-06-15"] * 4), "2020-01-01")
    assert np.allclose(np.linalg.norm(x[:, :3], axis=1), 1)
    assert np.allclose(x[0, :3], x[1, :3])  # every longitude at the pole is the same point
    km = gp.chord_to_km(np.linalg.norm(x[2, :3] - x[3, :3]))
    assert km == pytest.approx(0.2 * 111.19 * math.cos(math.radians(10)), rel=1e-3)  # across the dateline


def test_embed_time():
    x = gp.embed([0, 0], [0, 0], pd.to_datetime(["2020-12-31 12:00", "2021-01-01 12:00"]), "2020-01-01")
    assert gp.chord_to_days(np.linalg.norm(x[0, 3:5] - x[1, 3:5])) == pytest.approx(1.0, abs=0.01)
    assert x[1, 5] - x[0, 5] == pytest.approx(1 / 365.25)
    assert x[0, 5] == pytest.approx(365.5 / 365.25)


def test_unit_conversions_roundtrip():
    assert gp.chord_to_km(gp.km_to_chord(1234.0)) == pytest.approx(1234.0)
    assert gp.chord_to_days(gp.days_to_chord(45.0)) == pytest.approx(45.0)


@pytest.mark.parametrize("family,formula", [
    ("exponential", lambda r: math.exp(-r)),
    ("gaussian", lambda r: math.exp(-r * r / 2)),
    ("matern32", lambda r: (1 + math.sqrt(3) * r) * math.exp(-math.sqrt(3) * r)),
])
def test_kappa_matches_formula(family, formula):
    r = torch.tensor([0.0, 0.3, 1.0, 2.5], dtype=gp.DTYPE)
    assert np.allclose(gp.kappa(family, r ** 2).numpy(), [formula(float(v)) for v in r])


@pytest.mark.parametrize("family", gp.KERNELS)
def test_kernel_psd_and_variance(family):
    x = torch.as_tensor(random_inputs(300), dtype=gp.DTYPE)
    k = gp.kernel(family, x, x, P)
    assert torch.allclose(k.diagonal(), gp.kernel_diag(x, P))
    assert torch.allclose(k, k.T)
    assert torch.linalg.eigvalsh(k).min() > -1e-8 * k.diagonal().max()


@pytest.mark.parametrize("family", gp.KERNELS)
def test_vfe_with_all_inducing_points_is_exact(family):
    rng = np.random.default_rng(1)
    x = random_inputs(150, seed=2)
    xt = torch.as_tensor(x, dtype=gp.DTYPE)
    y = torch.as_tensor(rng.normal(size=150), dtype=gp.DTYPE)
    e = torch.as_tensor(rng.normal(size=150), dtype=gp.DTYPE)
    n = torch.as_tensor(rng.integers(1, 6, 150).astype(float), dtype=gp.DTYPE)
    model = gp.GP(family, {"jitter": 1e-10}, z=x)
    exact = gp.exact_log_marginal(family, P, xt, y, e, n)
    assert float(model.bound(P, xt, y, e, n)) == pytest.approx(float(exact), rel=1e-6)
    # predictions with Z = X equal the exact posterior
    xs = torch.as_tensor(random_inputs(40, seed=3, with_special=False), dtype=gp.DTYPE)
    es = torch.as_tensor(rng.normal(size=40), dtype=gp.DTYPE)
    m_ex, v_ex = gp.exact_predict(family, P, xt, y, e, n, xs, es)
    with torch.no_grad():
        l, lb, _, c, _, _ = model._terms(P, xt, y, e, n)
        kus = gp.kernel(family, model.z, xs, P)
        t1 = torch.linalg.solve_triangular(l, kus, upper=False)
        t2 = torch.linalg.solve_triangular(lb, t1, upper=False)
        m_vfe = P["beta0"] + P["beta1"] * es + t2.T @ c
        v_vfe = gp.kernel_diag(xs, P) - (t1 * t1).sum(0) + (t2 * t2).sum(0)
    assert torch.allclose(m_vfe, m_ex, atol=1e-6)
    assert torch.allclose(v_vfe, v_ex, atol=1e-6)


def test_vfe_bound_is_below_exact():
    rng = np.random.default_rng(4)
    x = random_inputs(200, seed=5)
    xt = torch.as_tensor(x, dtype=gp.DTYPE)
    y, e = (torch.as_tensor(rng.normal(size=200), dtype=gp.DTYPE) for _ in range(2))
    n = torch.ones(200, dtype=gp.DTYPE)
    model = gp.GP("matern32", {"jitter": 1e-10}, z=x[:30])
    assert float(model.bound(P, xt, y, e, n)) <= float(gp.exact_log_marginal("matern32", P, xt, y, e, n)) + 1e-8


def test_fit_recovers_mean_function_and_standardises_with_training_only():
    rng = np.random.default_rng(6)
    m = 400
    lat, lon = rng.uniform(-60, 60, m), rng.uniform(-180, 180, m)
    time = pd.Timestamp("2021-01-15") + pd.to_timedelta(rng.integers(0, 24, m) * 30, unit="D")
    x = gp.embed(lat, lon, time, "2020-01-01")
    e = rng.normal(230, 25, m)
    y = 20 + 0.8 * e + 5 * np.sin(np.radians(lon)) + rng.normal(0, 3, m)
    cfg = {"n_inducing": 60, "restarts": 1, "max_iter": 60, "seed": 0, "jitter": 1e-6,
           "init_sigma_A": 5.0, "init_sigma_B": 5.0, "init_ell_space_A_km": 2000.0, "init_ell_season_days": 60.0,
           "init_ell_space_B_km": 4000.0, "init_ell_time_years": 1.0, "init_sigma0": 5.0, "init_sigma1": 5.0}
    model = gp.GP("gaussian", cfg).fit(x[:300], y[:300], e[:300], np.ones(300))
    b0, b1, b2 = model.coefficients()
    assert b1 == pytest.approx(0.8, abs=0.1) and b2 == 0.0  # no gap step configured
    stats = dict(model.stats)
    pred = model.predict(x[300:], e[300:], np.ones(100))
    assert np.sqrt(np.mean((pred["mean"] - y[300:]) ** 2)) < 6
    assert np.all(pred["obs_var"] > pred["latent_var"])
    # test rows never touch the training statistics
    assert model.stats == stats and stats["y_mean"] == pytest.approx(y[:300].mean())
    # aggregate of one point equals its pointwise prediction
    agg_m, agg_v = model.aggregate(x[300:301], e[300:301], [1.0])
    assert agg_m == pytest.approx(pred["mean"][0]) and agg_v == pytest.approx(pred["latent_var"][0], rel=1e-6)


@pytest.mark.parametrize("family", gp.KERNELS)
def test_chunked_bound_and_gradient_equal_unchunked(family):
    rng = np.random.default_rng(7)
    x = torch.as_tensor(random_inputs(500, seed=8), dtype=gp.DTYPE)
    y, e = (torch.as_tensor(rng.normal(size=500), dtype=gp.DTYPE) for _ in range(2))
    n = torch.as_tensor(rng.integers(1, 4, 500).astype(float), dtype=gp.DTYPE)
    model = gp.GP(family, {"jitter": 1e-8}, z=x[:60].numpy())
    vals = {}
    for name, fn in (("full", lambda p: model.bound(p, x, y, e, n)),
                     ("chunked", lambda p: model.bound_chunked(p, x, y, e, n, 128))):
        raw = {k: v.clone().requires_grad_(True) for k, v in P.items()}
        b = fn(raw)
        b.backward()
        vals[name] = (float(b), torch.stack([raw[k].grad for k in sorted(raw)]))
    assert vals["chunked"][0] == pytest.approx(vals["full"][0], rel=1e-10)
    assert torch.allclose(vals["chunked"][1], vals["full"][1], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("family", gp.KERNELS)
def test_bound_with_gap_step_is_exact_and_chunks(family):
    rng = np.random.default_rng(9)
    x = random_inputs(120, seed=10)
    xt = torch.as_tensor(x, dtype=gp.DTYPE)
    y, e = (torch.as_tensor(rng.normal(size=120), dtype=gp.DTYPE) for _ in range(2))
    s = torch.as_tensor((rng.uniform(size=120) > 0.4).astype(float), dtype=gp.DTYPE)
    n = torch.as_tensor(rng.integers(1, 4, 120).astype(float), dtype=gp.DTYPE)
    p = P | {"beta2": torch.tensor(0.8, dtype=gp.DTYPE)}
    model = gp.GP(family, {"jitter": 1e-10, "gap_step": True}, z=x)
    exact = float(gp.exact_log_marginal(family, p, xt, y, e, n, s))
    assert float(model.bound(p, xt, y, e, n, s=s)) == pytest.approx(exact, rel=1e-6)
    assert float(model.bound_chunked(p, xt, y, e, n, 50, s)) == pytest.approx(exact, rel=1e-6)
    assert exact != pytest.approx(float(gp.exact_log_marginal(family, p, xt, y, e, n)), rel=1e-6)  # s is used


def test_fit_with_gap_step_recovers_the_step():
    rng = np.random.default_rng(11)
    m = 500
    lat, lon = rng.uniform(-60, 60, m), rng.uniform(-180, 180, m)
    time = pd.Timestamp("2020-01-15") + pd.to_timedelta(rng.integers(0, 81, m) * 30.44, unit="D")
    s = (time >= pd.Timestamp("2021-09-01")).astype(float)
    x = gp.embed(lat, lon, time, "2020-01-01")
    e = rng.normal(230, 25, m)
    y = 20 + 0.8 * e + 40 * s + 5 * np.sin(np.radians(lon)) + rng.normal(0, 3, m)
    cfg = {"n_inducing": 60, "restarts": 1, "max_iter": 60, "seed": 0, "jitter": 1e-6, "gap_step": True,
           "init_sigma_A": 5.0, "init_sigma_B": 5.0, "init_ell_space_A_km": 2000.0, "init_ell_season_days": 60.0,
           "init_ell_space_B_km": 4000.0, "init_ell_time_years": 1.0, "init_sigma0": 5.0, "init_sigma1": 5.0}
    model = gp.GP("matern32", cfg).fit(x, y, e, np.ones(m), s=s)
    b0, b1, b2 = model.coefficients()
    assert b2 == pytest.approx(40, abs=5) and b1 == pytest.approx(0.8, abs=0.1)
    assert model.hyperparameters()["beta2_wm2"] == b2
    with pytest.raises(ValueError):
        model.predict(x[:5], e[:5])  # a forgotten post_gap must not silently drop the step
    pre, post = (model.predict(x[:50], e[:50], s=np.full(50, v)) for v in (0.0, 1.0))
    assert np.allclose(post["mean"] - pre["mean"], b2) and np.allclose(post["latent_var"], pre["latent_var"])
    back = gp.GP.from_state(model.state())
    assert back.step and np.allclose(back.predict(x[:50], e[:50], s=np.ones(50))["mean"], post["mean"])
    # without the gap step, s is ignored and beta2 stays a fixed 0
    plain = gp.GP("matern32", dict(cfg, gap_step=False, max_iter=5)).fit(x, y, e, np.ones(m), s=s)
    assert plain.coefficients()[2] == 0.0 and float(plain.raw["beta2"]) == 0.0
