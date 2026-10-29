"""Fit the GP on every training row and save it, with its hyperparameters and in-sample fitted values.

With era5.radius_km = "auto", the ERA5 predictor is the smoothing radius whose ERA5 column correlates best with
CLARA on the training rows (the rule of the kernel comparison). The inducing inputs are k-means centres of the
training inputs; the fit maximises the VFE bound with L-BFGS from model.restarts fixed starts (gp.GP.fit).
"""

import time

import numpy as np
import pandas as pd
import torch

from . import gp, model_io
from .config import gp_settings


def set_threads(cfg):
    if cfg["model"]["threads"]:
        torch.set_num_threads(int(cfg["model"]["threads"]))


def choose_radius(obs, train, radii):
    """ERA5 smoothing radius with the highest correlation with CLARA on the training rows."""
    r = {rad: np.corrcoef(obs.clara[train], obs[f"era5_night_r{int(rad)}"][train])[0, 1] for rad in radii}
    return max(r, key=r.get), r


def era5_predictor(cfg, obs, train):
    """(column name, radius or None, correlations by radius) of the ERA5 predictor for these training rows."""
    radii = [r for r in cfg["era5"]["smoothing_radii_km"] if f"era5_night_r{int(r)}" in obs]
    if not radii:
        if "era5_night" not in obs:
            raise ValueError("no ERA5 predictor column (era5_night or era5_night_r<R>) in the training table")
        return "era5_night", None, {}
    choice = cfg["era5"]["radius_km"]
    if choice == "auto":
        radius, rs = choose_radius(obs, train, radii)
    else:
        radius, rs = float(choice), {}
        if f"era5_night_r{int(radius)}" not in obs:
            raise ValueError(f"era5.radius_km = {choice}, but the table has no era5_night_r{int(radius)} column")
    return f"era5_night_r{int(radius)}", float(radius), rs


def inputs(df, origin):
    return gp.embed(df.lat, df.lon, df.model_time, origin)


def fit(cfg, obs, train=None, z=None, verbose=True):
    """Fit model.kernel on the rows `train` (default all) of obs. Returns (model, era5 column, radius, correlations)."""
    train = np.ones(len(obs), bool) if train is None else np.asarray(train)
    column, radius, rs = era5_predictor(cfg, obs, train)
    x = inputs(obs, cfg["time"]["origin"])
    settings = gp_settings(cfg)
    if z is None:
        z = gp.inducing_inputs(x[train], settings["n_inducing"], settings, settings["seed"])
    s = obs.post_gap.to_numpy()
    model = gp.GP(cfg["model"]["kernel"], settings, z).fit(x[train], obs.clara[train], obs[column][train],
                                                            obs.n_overpasses[train], verbose=verbose, s=s[train])
    return model, column, radius, rs


def in_sample(model, obs, column, origin):
    x = inputs(obs, origin)
    pred = model.predict(x, obs[column].to_numpy(), obs.n_overpasses.to_numpy(), s=obs.post_gap.to_numpy())
    out = obs[["obs_id", "year", "month", "time", "lat", "lon", "clara", "n_overpasses", "post_gap",
               "cell_area_km2"]].copy()
    out["era5_night"] = obs[column].to_numpy()
    out["pred_mean"], out["mean_fn"] = pred["mean"], pred["mean_fn"]
    out["latent_sd"], out["pred_sd_obs"] = np.sqrt(pred["latent_var"]), np.sqrt(pred["obs_var"])
    out["residual"] = out.clara - out.pred_mean
    return out


def run(cfg, obs, out_dir, verbose=True):
    """Full-data fit, saved to out_dir: model.npz, model.json, hyperparameters.csv, insample_fitted.csv."""
    set_threads(cfg)
    if cfg["model"]["gap_step"] and obs.post_gap.nunique() < 2:
        raise ValueError(f"model.gap_step needs observations before and after time.gap_start "
                         f"({cfg['time']['gap_start']}); set --set model.gap_step=false otherwise")
    m = cfg["model"]
    print(f"fitting the {m['kernel']} GP on {len(obs):,} cell-months (M = {m['n_inducing']}, "
          f"{m['restarts']} start(s), up to {m['max_iter']} L-BFGS iterations each) ...", flush=True)
    t0 = time.time()
    model, column, radius, rs = fit(cfg, obs, verbose=verbose)
    seconds = time.time() - t0
    meta = {"res_deg": cfg["grid"]["res_deg"], "era5_column": column, "era5_radius_km": radius,
            "origin": cfg["time"]["origin"], "gap_start": cfg["time"]["gap_start"] if model.step else None,
            "n_train": len(obs), "fit_seconds": round(seconds, 1),
            "train_months": [str(obs.time.min())[:7], str(obs.time.max())[:7]],
            "era5_correlation_by_radius": {str(int(k)): float(v) for k, v in rs.items()}}
    model_io.save(model, out_dir, meta)
    hp = model.hyperparameters() | {"era5_radius_km": radius, "n_train": len(obs), "fit_seconds": seconds} \
        | {f"train_r_R{int(k)}": v for k, v in rs.items()}
    pd.DataFrame([hp]).to_csv(out_dir / "hyperparameters.csv", index=False, float_format="%.6g")
    fitted = in_sample(model, obs, column, cfg["time"]["origin"])
    fitted.to_csv(out_dir / "insample_fitted.csv", index=False, float_format="%.4f")
    rmse = float(np.sqrt(np.mean(fitted.residual ** 2)))
    b0, b1, b2 = model.coefficients()
    print(f"done in {seconds:.0f} s: beta0 {b0:.2f}, beta1 {b1:.4f}, beta2 {b2:.2f}, ERA5 radius "
          f"{'-' if radius is None else f'{radius:g} km'}, in-sample RMSE {rmse:.2f} W m-2 sr-1 -> {out_dir}")
    return model
