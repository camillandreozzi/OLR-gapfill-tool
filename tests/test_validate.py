"""Fold integrity of the blocked cross-validation and the metric definitions."""

import numpy as np
import pandas as pd
import pytest

from clara_gapfill import config, grid, validate

CFG = config.validate(config.defaults())


def synthetic_obs(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    lat_c, lon_c = grid.grid_centres(5)
    months = pd.date_range("2020-01-01", "2026-09-01", freq="MS")
    df = pd.DataFrame({"lat": rng.choice(lat_c, n).round(4), "lon": rng.choice(lon_c, n).round(4),
                       "time": rng.choice(months, n)}).drop_duplicates(["time", "lat", "lon"]).reset_index(drop=True)
    df.insert(0, "obs_id", np.arange(len(df)))
    return df


@pytest.fixture(scope="module")
def folds_and_obs():
    obs = synthetic_obs()
    return validate.make_folds(obs, CFG), obs


def test_roles_disjoint_and_test_partitions(folds_and_obs):
    folds, obs = folds_and_obs
    assert not folds.duplicated(["cv_type", "fold", "obs_id"]).any()  # test and buffer never overlap
    for t in ("spatial", "temporal"):
        test = folds[(folds.cv_type == t) & (folds.role == "test")]
        assert test.obs_id.is_unique and set(test.obs_id) == set(obs.obs_id)  # each obs tested exactly once
    assert (folds[folds.cv_type == "joint"].groupby("fold").role.apply(lambda r: (r == "test").sum()) >= 30).all()


def test_spatial_buffer_distance(folds_and_obs):
    folds, obs = folds_and_obs
    u = validate.unit_vectors(obs.lat, obs.lon)
    for fold, f in folds[folds.cv_type == "spatial"].groupby("fold"):
        test = np.isin(obs.obs_id, f.obs_id[f.role == "test"])
        train = ~np.isin(obs.obs_id, f.obs_id)
        d = validate.great_circle_km(u[train], u[test]).min(1)
        assert d.min() >= CFG["validate"]["spatial_buffer_km"] - 1e-6  # no training cell within the buffer


def test_temporal_buffer(folds_and_obs):
    folds, obs = folds_and_obs
    mi = (obs.time.dt.year * 12 + obs.time.dt.month).to_numpy()
    b = CFG["validate"]["temporal_buffer_months"]
    for fold, f in folds[folds.cv_type == "temporal"].groupby("fold"):
        test = np.isin(obs.obs_id, f.obs_id[f.role == "test"])
        train = ~np.isin(obs.obs_id, f.obs_id)
        assert np.abs(mi[train][:, None] - mi[test][None, :]).min() > b


def test_joint_halo(folds_and_obs):
    folds, obs = folds_and_obs
    u = validate.unit_vectors(obs.lat, obs.lon)
    mi = (obs.time.dt.year * 12 + obs.time.dt.month).to_numpy()
    for fold, f in folds[folds.cv_type == "joint"].groupby("fold"):
        test = np.isin(obs.obs_id, f.obs_id[f.role == "test"])
        train = ~np.isin(obs.obs_id, f.obs_id)
        close_t = np.abs(mi[train][:, None] - mi[test][None, :]) <= CFG["validate"]["joint_buffer_months"]
        close_s = validate.great_circle_km(u[train], u[test]) < CFG["validate"]["joint_buffer_km"]
        assert not (close_t & close_s).any()


def test_hash_stable_and_sensitive(folds_and_obs):
    folds, _ = folds_and_obs
    h = validate.folds_hash(folds)
    assert h == validate.folds_hash(folds.sample(frac=1, random_state=1))  # row order does not matter
    changed = folds.copy()
    changed.loc[0, "role"] = "buffer" if changed.loc[0, "role"] == "test" else "test"
    assert h != validate.folds_hash(changed)


def test_metrics_definitions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([1.5, 2.0, 2.0, 5.0])
    m = validate.metrics(y, mu, np.ones(4), np.array([1.0, 1.0, 1.0, 5.0]))
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 1) / 4))
    assert m["bias"] == pytest.approx(np.mean(mu - y))
    assert m["rmse_area"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 5) / 8))
    assert m["cover95"] == 1.0 and m["cover50"] == 0.5
    # CRPS of a Gaussian at its mean with sd 1: (sqrt2 - 1)/sqrt(pi)
    assert validate.crps_gaussian(0.0, 0.0, 1.0) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
