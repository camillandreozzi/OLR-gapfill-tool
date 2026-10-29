"""Blocked spatio-temporal cross-validation of the configured model (the scheme that selected the kernel).

Three validation types, each a leave-one-block-out test (never observation-level LOO):

    spatial   lat x lon blocks partitioning the globe. Training excludes the held-out block and a buffer:
              every cell whose centre lies within spatial_buffer_km (great circle) of a cell centre of the
              block, in every month.
    temporal  consecutive windows of temporal_window_months. Training excludes the window and
              temporal_buffer_months on either side, everywhere. Windows with fewer than MIN_TEST
              observations (e.g. inside the CLARA gap) are merged into the next one.
    joint     a spatial block over a time window. Training excludes the block during the window and the
              space-time halo: cells within joint_buffer_km of the block during the window +- joint_buffer_months.

Per fold, with training rows only: the ERA5 radius ("auto"), the standardisation, the inducing inputs and all
hyperparameters. The primary score is the mean over the validation types of the mean per-fold RMSE.
Finished folds are kept in validation/folds/ and skipped when the run is resumed.
"""

import hashlib
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import fit, grid

EARTH_RADIUS_KM = 6371.0
Z95 = stats.norm.ppf(0.975)
MIN_TEST = 30  # minimum held-out observations of a temporal or joint block


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def unit_vectors(lat, lon):
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def great_circle_km(u1, u2):
    """(n1, n2) great-circle distances between unit vectors."""
    return EARTH_RADIUS_KM * np.arccos(np.clip(u1 @ u2.T, -1, 1))


def _grid_cells(res):
    lat_c, lon_c = grid.grid_centres(res)
    lat, lon = np.meshgrid(lat_c, lon_c, indexing="ij")
    return lat.ravel().round(4), lon.ravel().round(4)


def spatial_blocks(block_lat, block_lon):
    """[(block_id, lat0, lat1, lon0, lon1)] windows with stride = window."""
    out = []
    for lat0 in np.arange(-90, 90, block_lat):
        for lon0 in np.arange(-180, 180, block_lon):
            lat1, lon1 = min(lat0 + block_lat, 90), min(lon0 + block_lon, 180)
            out.append((f"S{lat0:+04.0f}{lon0:+05.0f}", lat0, lat1, lon0, lon1))
    return out


def in_block(lat, lon, b):
    _, lat0, lat1, lon0, lon1 = b
    return (lat >= lat0) & (lat < lat1 if lat1 < 90 else lat <= 90) & (lon >= lon0) & (lon < lon1)


def block_distance(lat, lon, b, res):
    """Great-circle distance (km) from (lat, lon) points to the nearest grid-cell centre inside block b."""
    glat, glon = _grid_cells(res)
    inside = in_block(glat, glon, b)
    ilat, ilon = glat[inside], glon[inside]
    edge = (ilat == ilat.min()) | (ilat == ilat.max()) | (ilon == ilon.min()) | (ilon == ilon.max())
    pts = pd.DataFrame({"lat": np.round(lat, 4), "lon": np.round(lon, 4)})
    uniq = pts.drop_duplicates().reset_index(drop=True)
    d = np.empty(len(uniq))
    ue, uq = unit_vectors(ilat[edge], ilon[edge]), unit_vectors(uniq.lat, uniq.lon)
    for k in range(0, len(uniq), 20000):
        d[k:k + 20000] = great_circle_km(uq[k:k + 20000], ue).min(1)
    d[in_block(uniq.lat.to_numpy(), uniq.lon.to_numpy(), b)] = 0.0
    return pts.merge(uniq.assign(d=d), on=["lat", "lon"], how="left").d.to_numpy()


def month_windows(first, last, window):
    """[(block_id, start Period, end Period inclusive)] consecutive windows of `window` months."""
    months = pd.period_range(first, last, freq="M")
    return [(f"T{months[k]}", months[k], months[min(k + window, len(months)) - 1])
            for k in range(0, len(months), window)]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_folds(obs, cfg):
    """Fold table with columns cv_type, fold, block_id, obs_id, role (test / buffer); every other
    observation of the fold is training."""
    c, res = cfg["validate"], cfg["grid"]["res_deg"]
    lat, lon = obs.lat.to_numpy(), obs.lon.to_numpy()
    per = obs.time.dt.to_period("M")
    mi = (per.dt.year * 12 + per.dt.month).to_numpy()
    rows = []

    def add(cv_type, fold, block_id, test, buffer):
        buffer = buffer & ~test
        for role, mask in (("test", test), ("buffer", buffer)):
            ids = obs.obs_id.to_numpy()[mask]
            rows.append(pd.DataFrame({"cv_type": cv_type, "fold": fold, "block_id": block_id, "obs_id": ids,
                                      "role": role}))

    # spatial
    fold = 0
    for b in spatial_blocks(c["spatial_block_lat"], c["spatial_block_lon"]):
        test = in_block(lat, lon, b)
        if not test.any():
            continue
        add("spatial", fold, b[0], test, block_distance(lat, lon, b, res) < c["spatial_buffer_km"])
        fold += 1

    # temporal: a window with fewer than MIN_TEST observations is merged into the following window (the last
    # one into the previous), so no block score rests on a handful of cells
    wins = month_windows(cfg["time"]["first_month"], cfg["time"]["last_month"], c["temporal_window_months"])
    count = lambda w0, w1: int(((mi >= w0.year * 12 + w0.month) & (mi <= w1.year * 12 + w1.month)).sum())
    merged = []
    for wid, w0, w1 in wins:
        if merged and count(merged[-1][1], merged[-1][2]) < MIN_TEST:
            merged[-1] = (merged[-1][0], merged[-1][1], w1)
        else:
            merged.append((wid, w0, w1))
    if len(merged) > 1 and count(merged[-1][1], merged[-1][2]) < MIN_TEST:
        last = merged.pop()
        merged[-1] = (merged[-1][0], merged[-1][1], last[2])
    fold, bt = 0, c["temporal_buffer_months"]
    for wid, w0, w1 in merged:
        a, z = w0.year * 12 + w0.month, w1.year * 12 + w1.month
        test = (mi >= a) & (mi <= z)
        if not test.any():
            continue
        longer = z - a + 1 > c["temporal_window_months"]  # a merged window is labelled with its end
        add("temporal", fold, f"{wid}_{w1}" if longer else wid, test, (mi >= a - bt) & (mi <= z + bt))
        fold += 1

    # joint: deterministic sweep over (block, window) pairs with enough test observations
    jblocks = spatial_blocks(c["joint_block_lat"], c["joint_block_lon"])
    jwins = month_windows(cfg["time"]["first_month"], cfg["time"]["last_month"], c["joint_window_months"])
    bt, bkm = c["joint_buffer_months"], c["joint_buffer_km"]
    used, fold, k = set(), 0, 0
    nb, nw = len(jblocks), len(jwins)
    step = next(s for s in (7, 5, 11, 13, 3, 1) if np.gcd(s, nw) == 1)
    while fold < c["joint_n_folds"] and k < nb * nw:
        b, (wid, w0, w1) = jblocks[k % nb], jwins[(k * step) % nw]
        k += 1
        a, z = w0.year * 12 + w0.month, w1.year * 12 + w1.month
        sp = in_block(lat, lon, b)
        test = sp & (mi >= a) & (mi <= z)
        if test.sum() < MIN_TEST or (b[0], wid) in used:
            continue
        used.add((b[0], wid))
        near = sp | (block_distance(lat, lon, b, res) < bkm)
        add("joint", fold, f"{b[0]}_{wid}", test, near & (mi >= a - bt) & (mi <= z + bt))
        fold += 1
    return pd.concat(rows, ignore_index=True)


def folds_hash(folds):
    text = folds.sort_values(["cv_type", "fold", "role", "obs_id"]).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


def fold_summary(folds, n_obs):
    s = folds.groupby(["cv_type", "fold", "block_id", "role"]).size().unstack("role", fill_value=0)
    s["train"] = n_obs - s.get("test", 0) - s.get("buffer", 0)
    return s.reset_index()


def split(folds, cv_type, fold, obs_ids):
    """Boolean masks (train, test) over obs_ids for one fold."""
    f = folds[(folds.cv_type == cv_type) & (folds.fold == fold)]
    test = np.isin(obs_ids, f.obs_id[f.role == "test"])
    excluded = np.isin(obs_ids, f.obs_id)
    return ~excluded, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def metrics(y, mean, obs_var, area, mean_fn=None):
    """Metrics of predictions (mean, obs_var = predictive variance of a new cell-month mean) against y.
    area: cell areas for the area-weighted RMSE. bias = mean(prediction - observation)."""
    y, mean, obs_var, area = (np.asarray(v, float) for v in (y, mean, obs_var, area))
    sd = np.sqrt(obs_var)
    err = mean - y
    w = area / area.sum()
    out = {"n": len(y), "rmse": np.sqrt(np.mean(err ** 2)), "rmse_area": np.sqrt(w @ err ** 2),
           "mae": np.mean(np.abs(err)), "bias": np.mean(err),
           "nlpd": np.mean(0.5 * np.log(2 * np.pi * obs_var) + 0.5 * err ** 2 / obs_var),
           "crps": np.mean(crps_gaussian(y, mean, sd)), "width95": np.mean(2 * Z95 * sd),
           "r2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan}
    for level in (0.5, 0.8, 0.95):
        z = stats.norm.ppf(0.5 + level / 2)
        out[f"cover{int(level * 100)}"] = np.mean(np.abs(err) <= z * sd)
    if mean_fn is not None:
        e2 = np.asarray(mean_fn, float) - y
        out["rmse_mean_fn"] = np.sqrt(np.mean(e2 ** 2))
        out["r2_mean_fn"] = 1 - np.sum(e2 ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def fit_fold(cfg, obs, folds, cv_type, fold, out_dir, verbose=False):
    """Fit on the training rows of one fold, predict its test rows; resumable (one parquet per fold)."""
    path = out_dir / "folds" / f"{cv_type}_{fold:02d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train, test = split(folds, cv_type, fold, obs.obs_id.to_numpy())
    block_id = folds[(folds.cv_type == cv_type) & (folds.fold == fold)].block_id.iloc[0]
    t0 = time.time()
    model, column, radius, _ = fit.fit(cfg, obs, train, verbose=verbose)
    x = fit.inputs(obs[test], cfg["time"]["origin"])
    d = obs[test]
    pred = model.predict(x, d[column].to_numpy(), d.n_overpasses.to_numpy(), s=d.post_gap.to_numpy())
    frame = d[["obs_id", "year", "month", "time", "lat", "lon", "clara", "n_overpasses", "post_gap",
               "cell_area_km2"]].copy()
    frame["era5_night"], frame["era5_radius_km"] = d[column].to_numpy(), radius
    frame["pred_mean"], frame["mean_fn"] = pred["mean"], pred["mean_fn"]
    frame["latent_sd"], frame["pred_var_obs"] = np.sqrt(pred["latent_var"]), pred["obs_var"]
    frame["pred_sd_obs"] = np.sqrt(pred["obs_var"])
    frame["residual"] = frame.clara - frame.pred_mean
    frame["cv_type"], frame["fold"], frame["block_id"] = cv_type, fold, block_id
    frame["n_train"], frame["fit_seconds"] = int(train.sum()), time.time() - t0
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # the file appears only when complete (it marks the fold as done)
    return frame


def run(cfg, obs, out_dir, schemes=None, folds_only=None, verbose=False):
    """Cross-validate model.kernel; writes folds.csv, oos_predictions_<scheme>.csv, metrics_by_fold.csv,
    metrics_summary.csv to out_dir and returns the summary."""
    fit.set_threads(cfg)
    schemes = schemes or cfg["validate"]["schemes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(obs, cfg)
    h = folds_hash(folds)
    sha = out_dir / "folds_sha256.txt"
    if sha.exists() and sha.read_text().strip() != h and any((out_dir / "folds").glob("*.parquet")):
        raise RuntimeError(f"{out_dir} holds fold results made with other folds (data or [validate] changed); "
                           "delete that folder (or use another output_dir) first")
    folds.to_csv(out_dir / "folds.csv", index=False)
    sha.write_text(h + "\n")
    fold_summary(folds, len(obs)).to_csv(out_dir / "fold_summary.csv", index=False)
    rows, preds = [], {}
    for cv_type in schemes:
        ids = sorted(folds[folds.cv_type == cv_type].fold.unique())
        if folds_only is not None:
            ids = [f for f in ids if f in folds_only]
        frames = []
        for fold in ids:
            t0 = time.time()
            f = fit_fold(cfg, obs, folds, cv_type, int(fold), out_dir, verbose)
            m = metrics(f.clara, f.pred_mean, f.pred_var_obs, f.cell_area_km2, f.mean_fn)
            rows.append({"cv_type": cv_type, "fold": int(fold), "block_id": f.block_id.iloc[0]} | m)
            frames.append(f)
            print(f"  {cv_type} fold {fold:2d} {f.block_id.iloc[0]}: n_test {m['n']:5d}, RMSE {m['rmse']:6.2f}, "
                  f"cover95 {m['cover95']:.2f} ({time.time() - t0:.0f} s)", flush=True)
        if frames:
            preds[cv_type] = pd.concat(frames, ignore_index=True)
            preds[cv_type].to_csv(out_dir / f"oos_predictions_{cv_type}.csv", index=False, float_format="%.4f")
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(out_dir / "metrics_by_fold.csv", index=False, float_format="%.5g")
    summary = []
    for cv_type, p in preds.items():
        m = metrics(p.clara, p.pred_mean, p.pred_var_obs, p.cell_area_km2, p.mean_fn)
        summary.append({"cv_type": cv_type, "mean_fold_rmse": by_fold[by_fold.cv_type == cv_type].rmse.mean(),
                        "folds": int((by_fold.cv_type == cv_type).sum())} | {f"pooled_{k}": v for k, v in m.items()})
    summary = pd.DataFrame(summary)
    complete = folds_only is None and set(schemes) == set(cfg["validate"]["schemes"])
    if len(summary):
        primary = summary.mean_fold_rmse.mean()
        summary = pd.concat([summary, pd.DataFrame([{"cv_type": "primary_score" if complete else "mean_of_run",
                                                     "mean_fold_rmse": primary}])], ignore_index=True)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False, float_format="%.5g")
    return summary, preds.get("spatial")
