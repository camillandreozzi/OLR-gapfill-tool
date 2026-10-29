"""User-supplied tables: required columns, derived columns, and the gap-free prediction grid."""

import numpy as np
import pandas as pd
import pytest

from clara_gapfill import config, demo, predict, tables

CFG = config.load(None, ["grid.res_deg=10"])


def test_training_table_needs_its_columns():
    df = pd.DataFrame({"lat": [5.0], "lon": [5.0], "clara": [200.0]})
    with pytest.raises(ValueError) as err:
        tables.normalise(df, CFG, "training")
    for col in ("n_overpasses", "time (or year and month)", "era5_night"):
        assert col in str(err.value)


def test_training_table_derives_time_columns():
    df = pd.DataFrame({"lat": [5.0, 5.0], "lon": [15.0, 15.0], "year": [2021, 2022], "month": [8, 1],
                       "clara": [200.0, 210.0], "n_overpasses": [3, 2], "era5_night": [250.0, 255.0]})
    t = tables.normalise(df, CFG, "training")
    assert list(t.post_gap) == [0.0, 1.0] and list(t.period) == ["pre-gap", "gap"]
    assert list(t.time) == list(pd.to_datetime(["2021-08-01", "2022-01-01"]))
    assert t.model_time.iloc[0] == pd.Timestamp("2021-08-16 12:00")
    assert list(t.obs_id) == [0, 1] and (t.cell_area_km2 > 0).all()


def test_grid_table_from_dates_and_observed_flag():
    df = pd.DataFrame({"lat": [5.0, 5.0], "lon": [15.0, 25.0], "time": ["2023-03-17", "2023-03-02"],
                       "era5_night_r600": [250.0, 251.0], "clara": [np.nan, 199.0]})
    g = tables.normalise(df, CFG, "grid")
    assert list(g.clara_observed) == [False, True]
    assert (g.time == pd.Timestamp("2023-03-01")).all()


def test_demo_grid_is_complete_and_gap_free():
    cfg = config.load(None, ["grid.res_deg=30", "time.first_month='2020-01'", "time.last_month='2020-12'"])
    obs, g, _ = demo.synthetic_tables(cfg)
    assert len(g) == 12 * 6 * 12 and not g.duplicated(["time", "lat", "lon"]).any()
    assert g[[c for c in g if c.startswith("era5_night_r")]].notna().all().all()
    assert set(obs.obs_id) == set(g.obs_id[g.clara_observed])
    g["pred_mean"] = 1.0
    for c in predict.GRID_COLUMNS:
        if c not in g:
            g[c] = 0.0
    assert len(predict.grid_table(g, cfg)) == len(g)
    with pytest.raises(RuntimeError):
        predict.grid_table(g.iloc[1:], cfg)
