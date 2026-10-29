"""A synthetic end-to-end run, to check an installation without any CLARA or ERA5 data.

It writes a training table and a prediction grid in the format of the `tables` stage, with an ERA5-like field,
a known mean function (beta0 = 50, beta1 = 0.55, beta2 = 13), a smooth latent field, per-overpass noise, a
data gap and polar-day cells, then fits, predicts and maps with the normal stages (small settings by default).
"""

import numpy as np
import pandas as pd

from . import config, grid, pipeline, tables
from .layout import Layout

TRUTH = {"beta0": 50.0, "beta1": 0.55, "beta2": 13.0, "sigma1": 20.0}


def synthetic_tables(cfg, seed=0, observed_fraction=0.3):
    """(training table, prediction grid) on cfg's grid and months; also returns the true latent values."""
    rng = np.random.default_rng(seed)
    res, radii = cfg["grid"]["res_deg"], cfg["era5"]["smoothing_radii_km"]
    lat_c, lon_c = grid.grid_centres(res)
    months = config.months(cfg)
    t = pd.DatetimeIndex([m.to_timestamp() for m in months])
    T, LAT, LON = (a.ravel() for a in np.meshgrid(t, lat_c, lon_c, indexing="ij"))
    g = pd.DataFrame({"time": pd.to_datetime(T).astype("datetime64[ns]"), "lat": LAT, "lon": LON})
    phi, lam = np.radians(g.lat), np.radians(g.lon)
    season = np.cos(2 * np.pi * (g.time.dt.month - 1) / 12)
    years = (g.time - pd.Timestamp(cfg["time"]["origin"])).dt.days / 365.25
    base = 255 - 55 * np.sin(phi) ** 2 + 8 * np.cos(lam) * np.cos(phi) + 6 * np.sin(phi) * season
    for r in radii:  # finer radii carry more small-scale noise, so the largest radius correlates best
        g[f"era5_night_r{int(r)}"] = base + rng.normal(0, 4.0 / (1 + r / 300), len(g))
    g["era5_24h"] = base + 3
    summer = np.where(g.lat > 0, g.time.dt.month.isin([5, 6, 7, 8]), g.time.dt.month.isin([11, 12, 1, 2]))
    g["night_fallback"] = (np.abs(g.lat) > 66) & summer
    g["night_frac"] = np.where(g.night_fallback, 0.0, 0.4)
    g["model_time"] = tables.mid_month(g.time)
    g["post_gap"] = tables.post_gap_indicator(g.time, cfg["time"]["gap_start"])
    g["cell_area_km2"] = grid.cell_area_km2(g.lat, res)
    latent = (TRUTH["beta0"] + TRUTH["beta1"] * base + TRUTH["beta2"] * g.post_gap
              + 6 * np.sin(2 * lam) * np.cos(phi) + 4 * np.cos(2 * np.pi * years / 3))
    per = g.time.dt.to_period("M")
    in_gap = (per >= pd.Period(cfg["time"]["gap_start"], "M")) & (per <= pd.Period(cfg["time"]["gap_end"], "M"))
    observed = (rng.uniform(size=len(g)) < observed_fraction) & ~in_gap & ~g.night_fallback
    n = 1 + rng.poisson(3, len(g))
    clara = latent + rng.normal(0, 1, len(g)) * TRUTH["sigma1"] / np.sqrt(n)

    obs = g[observed].copy()
    obs["clara"], obs["n_overpasses"], obs["n_samples"] = clara[observed], n[observed], 5 * n[observed]
    obs["clara_sd"], obs["mean_sample_time"] = 30.0, obs.model_time
    obs["period"] = tables.period_label(obs.time, cfg["time"]["gap_start"], cfg["time"]["gap_end"])
    obs = obs.reset_index(drop=True)
    obs.insert(0, "obs_id", np.arange(len(obs)))
    obs.insert(1, "year", obs.time.dt.year)
    obs.insert(2, "month", obs.time.dt.month)

    g = g.merge(obs[["time", "lat", "lon", "obs_id", "clara", "n_samples", "n_overpasses"]],
                on=["time", "lat", "lon"], how="left")
    g["clara_observed"] = g.obs_id.notna()
    g["obs_id"] = g.obs_id.fillna(-1).astype(int)
    g["n_overpasses"] = g.n_overpasses.fillna(0).astype(int)
    g["n_samples"] = g.n_samples.fillna(0).astype(int)
    g.insert(0, "year", g.time.dt.year)
    g.insert(1, "month", g.time.dt.month)
    return obs, g, latent.to_numpy()


def demo_config(out, res, quick, overrides=()):
    sets = [f"paths.work_dir='{out / 'work'}'", f"paths.output_dir='{out / 'output'}'", f"grid.res_deg={res}",
            "time.first_month='2020-01'", "time.last_month='2023-12'",
            "era5.smoothing_radii_km=[0.0, 300.0, 600.0]",
            "model.n_inducing=60" if quick else "model.n_inducing=150",
            "model.restarts=1", "model.max_iter=40" if quick else "model.max_iter=100",
            "maps.months=['2020-07', '2022-01', '2023-07']" if quick else "maps.months='all'"]
    return config.load(None, sets + list(overrides))


def run(out, res=10, quick=False, overrides=()):
    out = out.resolve()
    cfg = demo_config(out, res, quick, overrides)
    lay = Layout(cfg)
    obs, g, latent = synthetic_tables(cfg)
    lay.work_res.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(lay.training_table, index=False)
    g.to_parquet(lay.prediction_grid, index=False)
    pipeline.write_manifest(lay.training_table, f"demo-{res}-{len(obs)}", observations=len(obs), grid_rows=len(g))
    print(f"demo: {len(obs):,} synthetic observed cell-months, {len(g):,} grid cell-months at {res} deg -> {lay.work_res}")
    pipeline.stage_fit(cfg, force=True, verbose=False)
    pred_dir = pipeline.stage_predict(cfg, force=True)
    pipeline.stage_maps(cfg, force=True)
    from . import model_io
    _, info = model_io.load(lay.model)
    hp = info["hyperparameters"]
    pred = pipeline.read_predictions(pred_dir).sort_values(["time", "lat", "lon"])
    truth = pd.DataFrame({"time": g.time, "lat": g.lat, "lon": g.lon, "latent": latent}).sort_values(["time", "lat", "lon"])
    unobserved = ~pred.clara_observed.to_numpy()
    rmse = float(np.sqrt(np.mean((pred.pred_mean.to_numpy() - truth.latent.to_numpy())[unobserved] ** 2)))
    print(f"demo: fitted beta0 {hp['beta0_wm2']:.1f}, beta1 {hp['beta1']:.3f}, beta2 {hp['beta2_wm2']:.1f} (simulated "
          f"{TRUTH['beta0']:g}, {TRUTH['beta1']:g}, {TRUTH['beta2']:g}; the GP takes up part of the smooth signal); "
          f"ERA5 radius {info['era5_radius_km']:g} km")
    print(f"demo: RMSE of the gap-filled field against the true field in unobserved cells {rmse:.2f} W m-2 sr-1 "
          f"(observation noise SD ~{TRUTH['sigma1'] / 2:g})")
    print(f"demo: outputs in {lay.out_res}")
    return info, rmse
