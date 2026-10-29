"""The model-ready tables on the grid.res_deg grid: CLARA cell-month means paired with ERA5, and the full grid.

Steps:
  1. Units and sign. CLARA OLR is a radiance used as stored (W m-2 sr-1, positive upward). ERA5 ttr (J m-2,
     negative upward) becomes OLR = -ttr / 3600 per hourly accumulation (era5.py). Physical units only; the
     model standardises internally with its training rows.
  2. Time. CLARA samples are binned by calendar month. The model time of a cell-month is the middle of the
     month (the estimand is the monthly mean), never the CLARA sample times.
  3. Space. ERA5 0.25 deg points -> 1 deg cell area means -> night-time mean -> spherical-cap smoothing at each
     era5.smoothing_radii_km -> res deg area means. CLARA footprints are binned into the res deg cells by
     position; no CLARA value is interpolated or spread to another cell.
  4. QC. Earth viewing, plausible range, night side (SZA > qc.night_sza_min), VZA <= qc.vza_max (samples.py).
  5. Pairing. Each observed cell-month (>= 1 QC'd sample) gets the ERA5 night-time OLR of the same cell and
     month. The prediction grid holds every cell of every month with ERA5, with a clara_observed flag.
  6. Regime. post_gap = 1 from time.gap_start on, 0 before: the predictor of the calibration step across the
     CLARA gap. The gap months, never observed, count as post-gap.

Training table columns: obs_id, year, month, time (month start), lat, lon (cell centre), clara, clara_sd,
n_samples, n_overpasses, mean_sample_time, era5_night_r<R> (one per radius), era5_24h, night_frac,
night_fallback, model_time, cell_area_km2, period, post_gap.
Prediction grid columns: year, month, time, lat, lon, era5_night_r<R>, era5_24h, night_frac, night_fallback,
obs_id (-1 = not observed), clara, n_samples, n_overpasses, clara_observed, model_time, post_gap, cell_area_km2.
"""

import numpy as np
import pandas as pd

from . import era5_night, grid, samples
from .config import months as config_months


def post_gap_indicator(time, gap_start):
    """1.0 for months from gap_start on (the gap months included), 0.0 before."""
    per = pd.Series(pd.to_datetime(time)).dt.to_period("M")
    return (per >= pd.Period(gap_start, "M")).astype(float).to_numpy()


def mid_month(month_start):
    """Middle of the month of each month-start timestamp (Series or array)."""
    t = pd.Series(pd.to_datetime(month_start))
    return (t + pd.to_timedelta(t.dt.days_in_month / 2, unit="D")).to_numpy()


def period_label(time, gap_start, gap_end):
    per = pd.Series(pd.to_datetime(time)).dt.to_period("M")
    gs, ge = pd.Period(gap_start, "M"), pd.Period(gap_end, "M")
    return np.where(per < gs, "pre-gap", np.where(per > ge, "post-gap", "gap"))


def _ns(t):
    return pd.to_datetime(t).astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# CLARA cell-months and ERA5 on the grid
# ---------------------------------------------------------------------------

def bin_cell_months(q, res):
    """Cell-month table of QC'd CLARA samples q: lat, lon (cell centre), time (month start), clara (mean),
    clara_sd (within-cell sample SD), n_samples, n_overpasses, mean_sample_time."""
    lat_c, lon_c = grid.grid_centres(res)
    i, j = grid.cell_index(q.clara_lat.to_numpy(), q.clara_lon.to_numpy(), res)
    df = pd.DataFrame({"time": q.time.dt.to_period("M").dt.to_timestamp().to_numpy(), "i": i, "j": j,
                       "olr": q.olr.to_numpy(), "overpass": q.overpass.to_numpy(),
                       "t": q.time.to_numpy()})
    g = df.groupby(["time", "i", "j"], sort=True)
    out = g.agg(clara=("olr", "mean"), clara_sd=("olr", "std"), n_samples=("olr", "size"),
                n_overpasses=("overpass", "nunique"), mean_sample_time=("t", "mean")).reset_index()
    out.insert(1, "lat", lat_c[out.i])
    out.insert(2, "lon", lon_c[out.j])
    return out


def era5_on_grid(night, res, radii_km):
    """ERA5 night-time OLR (smoothed at each radius) and 24 h OLR as res deg cell-month area means."""
    sm = era5_night.smoothed(night, "olr_night", radii_km)
    sm["olr_24h"] = night.olr_24h
    sm["night_frac"] = night.night_frac
    sm["night_fallback"] = night.night_fallback.astype(float)  # > 0 where (part of) the cell is in polar day
    return grid.coarsen(sm, res)


def grid_to_table(ds, names):
    """(time, lat, lon) Dataset -> long table with the given variables, coordinates rounded for joins."""
    df = ds[list(names)].to_dataframe().reset_index()
    df["lat"], df["lon"] = df.lat.round(4), df.lon.round(4)
    df["time"] = _ns(df.time)
    return df


def build(cfg, geolocated, night):
    """(flagged samples, training table, prediction grid) from the geolocated CLARA samples and the 1 deg
    ERA5 night-time product."""
    res = cfg["grid"]["res_deg"]
    radii = cfg["era5"]["smoothing_radii_km"]
    t = cfg["time"]
    first, last = pd.Timestamp(t["first_month"]), pd.Timestamp(t["last_month"])

    s = samples.add_flags(geolocated, cfg["qc"])
    q = s[s.qc]
    cells = bin_cell_months(q, res)
    cells["lat"], cells["lon"] = cells.lat.round(4), cells.lon.round(4)
    cells["time"] = _ns(cells.time)
    cells = cells[(cells.time >= first) & (cells.time <= last)]

    era = era5_on_grid(night, res, radii)
    names = [f"olr_night_r{int(r)}" for r in radii] + ["olr_24h", "night_frac", "night_fallback"]
    era_t = grid_to_table(era, names).rename(columns={f"olr_night_r{int(r)}": f"era5_night_r{int(r)}" for r in radii}
                                             | {"olr_24h": "era5_24h"})

    obs = cells.merge(era_t, on=["time", "lat", "lon"], how="left")
    no_era5 = obs[f"era5_night_r{int(radii[0])}"].isna()
    if no_era5.any():
        months = sorted(obs.time[no_era5].dt.strftime("%Y-%m").unique())
        print(f"warning: {int(no_era5.sum())} observed cell-months have no ERA5 (months {', '.join(months)}); "
              "they are left out of the training table")
        obs = obs[~no_era5]
    obs["night_fallback"] = obs.night_fallback > 0
    obs["model_time"] = mid_month(obs.time)
    obs["cell_area_km2"] = grid.cell_area_km2(obs.lat, res)
    obs["period"] = period_label(obs.time, t["gap_start"], t["gap_end"])
    obs["post_gap"] = post_gap_indicator(obs.time, t["gap_start"])
    obs = obs.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
    obs.insert(0, "obs_id", np.arange(len(obs)))
    obs.insert(1, "year", obs.time.dt.year)
    obs.insert(2, "month", obs.time.dt.month)
    obs = obs.drop(columns=["i", "j"])

    gridt = era_t[(era_t.time >= first) & (era_t.time <= last)].copy()
    gridt = gridt.merge(obs[["time", "lat", "lon", "obs_id", "clara", "n_samples", "n_overpasses"]],
                        on=["time", "lat", "lon"], how="left")
    gridt["clara_observed"] = gridt.obs_id.notna()
    gridt["obs_id"] = gridt.obs_id.fillna(-1).astype(int)
    gridt["n_overpasses"] = gridt.n_overpasses.fillna(0).astype(int)
    gridt["n_samples"] = gridt.n_samples.fillna(0).astype(int)
    gridt["night_fallback"] = gridt.night_fallback > 0
    gridt["model_time"] = mid_month(gridt.time)
    gridt["post_gap"] = post_gap_indicator(gridt.time, t["gap_start"])
    gridt["cell_area_km2"] = grid.cell_area_km2(gridt.lat, res)
    gridt.insert(0, "year", gridt.time.dt.year)
    gridt.insert(1, "month", gridt.time.dt.month)
    gridt = gridt.reset_index(drop=True)
    present = set(pd.DatetimeIndex(gridt.time.unique()).to_period("M"))
    missing = [str(p) for p in config_months(cfg) if p not in present]
    if missing:
        print(f"warning: no ERA5 for {', '.join(missing)}; these months are not in the prediction grid")
    return s, obs, gridt


def validation_checks(obs, gridt, cfg):
    """Sanity checks of the two tables, with the expected values."""
    w = np.cos(np.radians(gridt.lat))
    res = cfg["grid"]["res_deg"]
    n_cells = (180 // res) * (360 // res)
    era5_cols = [c for c in gridt.columns if c.startswith("era5_night_r")]
    checks = [
        ("ERA5 global mean OLR, 24 h (W m-2)", np.average(gridt.era5_24h, weights=w), "~240"),
        ("ERA5 global mean OLR, night-time (fallback in polar day) (W m-2)",
         np.average(gridt[era5_cols[0]], weights=w), "close to the 24 h mean"),
        ("observed cell-months", len(obs), ">0"),
        ("observed cell-months in polar-day fallback cells", int(obs.night_fallback.sum()), "~0"),
        ("cell-months in prediction grid", len(gridt),
         f"{gridt.time.nunique()} months x {n_cells:,} cells = {gridt.time.nunique() * n_cells:,}"),
        ("prediction-grid cell-months without a predictor value (gaps)",
         int(gridt[era5_cols].isna().any(axis=1).sum()), "0"),
        ("prediction-grid cell-months in polar day (darkest-hours fallback)", int(gridt.night_fallback.sum()),
         "flagged, extrapolation"),
        ("duplicate (time, lat, lon) in observations", int(obs.duplicated(["time", "lat", "lon"]).sum()), "0"),
        ("observed months before / from time.gap_start",
         f"{obs.time[obs.post_gap == 0].nunique()} / {obs.time[obs.post_gap == 1].nunique()}",
         ">0 / >0 (needed for model.gap_step)"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


def observations_summary(obs):
    return obs.groupby("period").agg(cell_months=("clara", "size"), months=("time", "nunique"),
                                     median_samples=("n_samples", "median"),
                                     median_overpasses=("n_overpasses", "median"),
                                     clara_mean=("clara", "mean"),
                                     era5_night_mean=(next(c for c in obs if c.startswith("era5_night_r")), "mean"))


# ---------------------------------------------------------------------------
# Tables brought by the user (clara-gapfill fit --table / predict --grid)
# ---------------------------------------------------------------------------

def read_table(path):
    path = str(path)
    return pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)


def era5_columns(df):
    return [c for c in df.columns if c.startswith("era5_night_r") or c == "era5_night"]


def normalise(df, cfg, kind):
    """Check a user table and derive the columns the model needs.

    kind "training": lat, lon, clara, n_overpasses, a month (time, or year and month) and ERA5 night-time OLR
    (era5_night, or era5_night_r<R> columns). kind "grid": the same without clara and n_overpasses (clara, if
    present, gives the observed values for residuals). time, model_time, post_gap, cell_area_km2, period,
    night_fallback, clara_observed and obs_id are derived when missing.
    """
    df = df.copy()
    need = ["lat", "lon"] + (["clara", "n_overpasses"] if kind == "training" else [])
    missing = [c for c in need if c not in df]
    if "time" not in df and not {"year", "month"} <= set(df):
        missing.append("time (or year and month)")
    if not era5_columns(df):
        missing.append("era5_night (or era5_night_r<R>)")
    if missing:
        raise ValueError(f"the {kind} table lacks the columns: {', '.join(missing)}")
    if "time" in df:
        df["time"] = _ns(pd.to_datetime(df.time).dt.to_period("M").dt.to_timestamp())
    else:
        df["time"] = _ns(pd.to_datetime(pd.DataFrame({"year": df.year, "month": df.month, "day": 1})))
    df["year"], df["month"] = df.time.dt.year, df.time.dt.month
    t, res = cfg["time"], cfg["grid"]["res_deg"]
    if "model_time" not in df:
        df["model_time"] = mid_month(df.time)
    if "post_gap" not in df:
        df["post_gap"] = post_gap_indicator(df.time, t["gap_start"])
    if "cell_area_km2" not in df:
        df["cell_area_km2"] = grid.cell_area_km2(df.lat, res)
    if "period" not in df:
        df["period"] = period_label(df.time, t["gap_start"], t["gap_end"])
    df["night_fallback"] = df.night_fallback.astype(bool) if "night_fallback" in df else False
    if kind == "training":
        bad = df[["lat", "lon", "clara", "n_overpasses"] + era5_columns(df)].isna().any(axis=1)
        if bad.any():
            print(f"warning: {int(bad.sum())} training rows with missing values are left out")
            df = df[~bad]
        if (df.n_overpasses <= 0).any():
            raise ValueError("n_overpasses must be >= 1 on every training row")
        df = df.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
        df["obs_id"] = np.arange(len(df))
    else:
        if "clara" not in df:
            df["clara"] = np.nan
        df["clara_observed"] = df.clara_observed.astype(bool) if "clara_observed" in df else df.clara.notna()
        if "n_overpasses" not in df:
            df["n_overpasses"] = 0
        if "obs_id" not in df:
            df["obs_id"] = -1
        df = df.reset_index(drop=True)
    return df
