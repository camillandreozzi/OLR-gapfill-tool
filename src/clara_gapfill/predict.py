"""Gap-filled predictions on the prediction grid from a fitted model.

Each prediction is the monthly-mean CLARA night-side OLR of a cell (W m-2 sr-1). Two uncertainties:

    latent_sd     SD of the GP posterior of the monthly-mean CLARA OLR (what the uncertainty maps show)
    pred_sd_obs   SD for a new noisy CLARA cell-month mean from predict.n_overpasses_for_sd overpasses
                  = sqrt(latent_var + sigma0^2 + sigma1^2 / n)

Residuals are observed - predicted where CLARA observed the cell-month: in-sample (the observation was used in
the fit) and, after clara-gapfill validate, out-of-sample from spatial block CV.

Polar-day cells (no night that month) use the darkest-hours ERA5 fallback and are never observed by CLARA, so
their predictions are extrapolation (night_fallback). The months of the CLARA gap are never observed either;
with the gap step they take the post-gap level (post_gap = 1).
"""

import numpy as np
import pandas as pd
import xarray as xr

from . import gp, grid

GRID_COLUMNS = ["year", "month", "time", "lat", "lon", "pred_mean", "latent_sd", "pred_sd_obs", "mean_fn",
                "era5_night", "post_gap", "clara_observed", "clara", "n_overpasses", "night_fallback",
                "residual_insample", "residual_oos_spatial", "cell_area_km2"]


def check_compatible(info, cfg, user_grid=False):
    """Refuse a grid the model was not fitted for: other resolution or another gap definition."""
    if not user_grid and info["res_deg"] != cfg["grid"]["res_deg"]:
        raise ValueError(f"the model was fitted at {info['res_deg']} deg, the grid is {cfg['grid']['res_deg']} deg; "
                         f"use --set grid.res_deg={info['res_deg']} or fit a model at this resolution")
    if info.get("gap_step") and info.get("gap_start") and info["gap_start"] != cfg["time"]["gap_start"]:
        raise ValueError(f"the model's gap step starts at {info['gap_start']}, the config's time.gap_start is "
                         f"{cfg['time']['gap_start']}: post_gap would mean something else")


def era5_column(info, gridt):
    col = info["era5_column"]
    if col in gridt:
        return col
    if "era5_night" in gridt:
        if info.get("era5_radius_km") is not None:
            print(f"note: the grid has no {col} column; using its era5_night column (it should be ERA5 night-time "
                  f"OLR smoothed over {info['era5_radius_km']:g} km caps, as in the fit)")
        return "era5_night"
    raise ValueError(f"the prediction grid has no {col} (or era5_night) column")


def predict_grid(model, info, gridt, n_sd=1.0, oos=None, batch=5000):
    """The grid with pred_mean, mean_fn, latent_sd, pred_sd_obs, era5_night and the residual columns."""
    col = era5_column(info, gridt)
    e = gridt[col].to_numpy(float)
    if not np.isfinite(e).all():
        raise ValueError(f"{(~np.isfinite(e)).sum()} prediction-grid cell-months have no ERA5 predictor; "
                         "the grid must be gap-free")
    x = gp.embed(gridt.lat, gridt.lon, gridt.model_time, info["origin"])
    p = model.predict(x, e, np.full(len(e), float(n_sd)), batch=int(batch), s=gridt.post_gap.to_numpy())
    g = gridt.copy()
    g["pred_mean"], g["mean_fn"] = p["mean"], p["mean_fn"]
    g["latent_sd"], g["pred_sd_obs"] = np.sqrt(p["latent_var"]), np.sqrt(p["obs_var"])
    g["era5_night"] = e
    g["residual_insample"] = np.where(g.clara_observed, g.clara - g.pred_mean, np.nan)
    g["residual_oos_spatial"] = np.nan
    if oos is not None and len(oos):
        key = ["time", "lat", "lon"]
        o = oos[key + ["residual"]].rename(columns={"residual": "residual_oos_spatial"})
        g = g.drop(columns="residual_oos_spatial").merge(o, on=key, how="left")
        g.loc[~g.clara_observed.astype(bool), "residual_oos_spatial"] = np.nan
    return g, x


def to_dataset(g, res):
    """(time, lat, lon) Dataset of the prediction columns of the grid rows g."""
    lat_c, lon_c = grid.grid_centres(res)
    months = sorted(g.time.unique())
    shape = (len(months), len(lat_c), len(lon_c))
    ti = pd.DatetimeIndex(months).get_indexer(g.time)
    i, j = grid.cell_index(g.lat.to_numpy(), g.lon.to_numpy(), res)
    ds = xr.Dataset(coords={"time": months, "lat": lat_c, "lon": lon_c})
    spec = {"pred_mean": ("W m-2 sr-1", "GP posterior mean of monthly-mean CLARA night-side OLR"),
            "mean_fn": ("W m-2 sr-1", "mean function beta0 + beta1 ERA5 + beta2 post_gap of the prediction"),
            "latent_sd": ("W m-2 sr-1", "posterior SD of the monthly-mean CLARA OLR (latent GP)"),
            "pred_sd_obs": ("W m-2 sr-1", "predictive SD of a new CLARA cell-month mean (predict.n_overpasses_for_sd)"),
            "clara": ("W m-2 sr-1", "observed CLARA cell-month mean OLR, a radiance (QC'd samples)"),
            "n_overpasses": ("1", "CLARA overpasses behind the observation"),
            "residual_insample": ("W m-2 sr-1", "observed - predicted, full-data fit (in-sample)"),
            "residual_oos_spatial": ("W m-2 sr-1", "observed - predicted, spatial block CV (out-of-sample)"),
            "era5_night": ("W m-2", "ERA5 night-time OLR used as predictor (darkest hours where night_fallback)"),
            "post_gap": ("1", "1 from time.gap_start on (post-gap calibration regime), 0 before"),
            "clara_observed": ("1", "1 where CLARA observed the cell-month"),
            "night_fallback": ("1", "1 where the cell has no night that month (polar day): extrapolation")}
    for v, (units, name) in spec.items():
        a = np.full(shape, np.nan)
        a[ti, i, j] = g[v].astype(float).to_numpy()
        ds[v] = (("time", "lat", "lon"), a)
        ds[v].attrs = {"units": units, "long_name": name}
    return ds


def global_mean(model, g, x):
    """Area-weighted global mean per month with the posterior SD from the full covariance (and, for comparison,
    the SDs the cells would give if they were independent or perfectly correlated)."""
    rows = []
    e, s = g.era5_night.to_numpy(), g.post_gap.to_numpy()
    area = g.cell_area_km2.to_numpy()
    groups = g.groupby("time").indices
    # within one month every cell has the same time coordinates, so w^T K w is the same for every month
    first = np.asarray(next(iter(groups.values())))
    prior_quad = model.prior_quadratic(x[first], area[first])
    for t, idx in groups.items():
        idx = np.asarray(idx)
        w = area[idx]
        same_cells = len(idx) == len(first)
        m, v = model.aggregate(x[idx], e[idx], w, prior_quad if same_cells else None, s=s[idx])
        wn = w / w.sum()
        lv = g.latent_sd.to_numpy()[idx] ** 2
        fb = g.night_fallback.to_numpy()[idx].astype(float)
        rows.append({"time": t, "global_mean": m, "sd_full_covariance": np.sqrt(v),
                     "sd_if_independent": np.sqrt(wn ** 2 @ lv), "sd_mean_of_variances": np.sqrt(wn @ lv),
                     "polar_day_area_fraction": wn @ fb, "cells": len(idx)})
    return pd.DataFrame(rows)


def grid_table(g, cfg, complete=True):
    """Every cell of every month in one table, sorted; with complete, checked to be the full lat x lon x month
    product of the months present."""
    res = cfg["grid"]["res_deg"]
    table = g[GRID_COLUMNS].sort_values(["time", "lat", "lon"]).astype({"clara_observed": int, "night_fallback": int})
    if complete:
        expected = g.time.nunique() * (180 // res) * (360 // res)
        if len(table) != expected or table.duplicated(["time", "lat", "lon"]).any() or table.pred_mean.isna().any():
            raise RuntimeError(f"prediction table has {len(table):,} rows, expected {expected:,} unique cell-months "
                               "with a prediction each")
    return table


def write(g, x, model, info, cfg, out_dir, complete=True):
    """Write the grid table, the yearly NetCDF files, the monthly CSVs and the global mean (per [predict])."""
    pc = cfg["predict"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res = cfg["grid"]["res_deg"]
    table = grid_table(g, cfg, complete)
    path = out_dir / f"grid_predictions.{pc['table_format']}"
    for old in out_dir.glob("grid_predictions.*"):
        old.unlink()
    if pc["table_format"] == "parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False, float_format="%.3f")
    print(f"wrote {len(table):,} cell-months to {path}")
    attrs = {"kernel": info["kernel"], "era5_column": info["era5_column"], "res_deg": res,
             "estimand": "monthly-mean CLARA night-side OLR (radiance, W m-2 sr-1)", "source": "clara-gapfill"}
    for year, gy in g.groupby("year"):
        if pc["netcdf"]:
            ds = to_dataset(gy, res)
            ds.attrs = attrs
            (out_dir / "netcdf").mkdir(exist_ok=True)
            ds.to_netcdf(out_dir / "netcdf" / f"{year}_predictions.nc")
        if pc["monthly_csv"]:
            (out_dir / "monthly_csv").mkdir(exist_ok=True)
            for t, gm in gy.groupby("time"):
                gm[GRID_COLUMNS].to_csv(out_dir / "monthly_csv" / f"{pd.Timestamp(t):%Y_%m}_predictions.csv",
                                        index=False, float_format="%.3f")
    gm = None
    if pc["global_mean"] and complete:
        gm = global_mean(model, g, x)
        gm.to_csv(out_dir / "global_mean_monthly.csv", index=False, float_format="%.4f")
        print(f"wrote the monthly global mean to {out_dir / 'global_mean_monthly.csv'}")
    return table, gm
