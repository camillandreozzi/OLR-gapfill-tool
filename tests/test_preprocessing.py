"""Unit and sign conversions, grid conventions, binning, QC, solar geometry, overpasses, ERA5 night weights
and cap smoothing."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clara_gapfill import era5, era5_night, grid, samples, solar, tables

QC = {"olr_min": 50.0, "olr_min_from": {2024: 75.0, 2025: 90.0}, "olr_max": 400.0, "night_sza_min": 100.0,
      "vza_max": 60.0, "overpass_gap_min": 20.0}


def test_era5_by_hour_sign_units_and_months(tmp_path):
    # 240 W m-2 upward over one hour = -240 * 3600 J m-2 in ERA5's convention; stamps 01:00 day 1 .. 00:00 day 2
    stamps = pd.date_range("2021-03-01 01:00", periods=24, freq="h").append(
        pd.date_range("2021-04-01 01:00", periods=24, freq="h"))
    ds = xr.Dataset({"ttr": (("valid_time", "latitude", "longitude"), np.full((48, 3, 4), -240.0 * 3600))},
                    coords={"valid_time": stamps, "latitude": [90.0, 0.0, -90.0],
                            "longitude": [0.0, 90.0, 180.0, 270.0]})
    ds.to_netcdf(tmp_path / "x.nc")
    blocks = list(era5.month_blocks(tmp_path / "x.nc"))
    assert [str(m) for m, _ in blocks] == ["2021-03", "2021-04"]  # the 00:00 stamp of day 2 belongs to March
    out = blocks[0][1]
    assert np.allclose(out.olr.values, 240.0)
    assert list(out.lat.values) == [-90.0, 0.0, 90.0]  # ascending
    assert list(out.lon.values) == [-180.0, -90.0, 0.0, 90.0]  # 0-360 -> [-180, 180), sorted
    assert [str(m) for m, _ in era5.month_blocks(tmp_path / "x.nc", {pd.Period("2021-04", "M")})] == ["2021-04"]


def test_era5_requests_only_published_months(tmp_path):
    months = pd.period_range("2025-11", "2026-02", freq="M")
    reqs = era5.requests(months, tmp_path, today=pd.Timestamp("2026-02-05").date())
    assert [(p.name, r["month"]) for p, r in reqs] == [("era5_ttr_monthly_by_hour_2025.nc", ["11", "12"])]


def test_era5_regrid_to_1deg_keeps_constant_and_area_mean():
    lat, lon = np.linspace(-90, 90, 721), np.arange(-180, 180, 0.25)
    rng = np.random.default_rng(0)
    v = 240 + rng.normal(0, 1, (1, 721, 1440))
    ds = xr.Dataset({"olr": (("time", "lat", "lon"), v)}, coords={"time": [pd.Timestamp("2021-01-01")],
                                                                  "lat": lat, "lon": lon})
    one = grid.era5_to_1deg(ds)
    assert one.olr.shape == (1, 180, 360)
    const = grid.era5_to_1deg(ds.assign(olr=ds.olr * 0 + 7.0))
    assert np.allclose(const.olr.values, 7.0)
    with pytest.raises(ValueError):
        grid.era5_to_1deg(ds.isel(lat=slice(0, 720)))


def test_coarsen_preserves_area_mean():
    lat, lon = np.arange(-89.5, 90), np.arange(-179.5, 180)
    rng = np.random.default_rng(0)
    v = rng.normal(240, 30, (1, 180, 360))
    ds = xr.Dataset({"olr": (("time", "lat", "lon"), v)}, coords={"time": [pd.Timestamp("2021-01-01")],
                                                                  "lat": lat, "lon": lon})
    c = grid.coarsen(ds, 5)
    w1 = np.cos(np.radians(lat))[:, None] * np.ones(360)
    w5 = np.cos(np.radians(lat)).reshape(36, 5).sum(1)[:, None] * np.ones(72)
    assert np.average(c.olr.values[0], weights=w5) == pytest.approx(np.average(v[0], weights=w1), rel=1e-12)


def test_cell_index_edges_poles_dateline():
    i, j = grid.cell_index([-90, 90, 0, 0, 4.999, 5.0], [-180, 180, -0.001, 0, 0, 0], 5)
    assert list(i) == [0, 35, 18, 18, 18, 19]
    assert list(j) == [0, 71, 35, 36, 36, 36]


@pytest.mark.parametrize("res", [1, 5, 10])
def test_cell_area_sums_to_sphere(res):
    lat_c, lon_c = grid.grid_centres(res)
    total = grid.cell_area_km2(lat_c, res).sum() * len(lon_c)
    assert total == pytest.approx(4 * np.pi * 6371.0 ** 2, rel=1e-12)


def test_bin_cell_months_and_overpasses():
    t = pd.to_datetime(["2021-01-03 00:00:00", "2021-01-03 00:00:30", "2021-01-03 00:01:00",  # one overpass
                        "2021-01-03 01:40:00",  # next orbit, same cell
                        "2021-02-01 00:00:00"])
    q = pd.DataFrame({"time": t, "clara_lat": [1.0, 1.5, 2.0, 3.0, 1.0], "clara_lon": [1.0, 1.2, 1.4, 2.0, 1.0],
                      "olr": [200.0, 210.0, 220.0, 260.0, 180.0]})
    q["overpass"] = samples.overpass_ids(q.time, 20.0)
    assert list(q.overpass) == [0, 0, 0, 1, 2]
    cells = tables.bin_cell_months(q, 5)
    jan = cells[cells.time == pd.Timestamp("2021-01-01")].iloc[0]
    assert (jan.lat, jan.lon) == (2.5, 2.5)
    assert jan.clara == pytest.approx(222.5) and jan.n_samples == 4 and jan.n_overpasses == 2
    assert len(cells) == 2


def test_in_range_limit_rises_by_year():
    t = pd.Series(pd.to_datetime(["2023-06-01", "2024-06-01", "2025-06-01", "2025-06-01"]))
    assert list(samples.in_range(np.array([60.0, 60.0, 85.0, 450.0]), t, QC)) == [True, False, False, False]


def test_add_flags_qc_and_overpasses():
    df = pd.DataFrame({"time": pd.to_datetime(["2021-03-20 00:00:00", "2021-03-20 00:00:30", "2021-03-20 12:00:00",
                                               "2021-03-20 00:01:00"]),
                       "olr": [200.0, 210.0, 220.0, 1e4], "clara_lat": [0.0, 0.0, 0.0, 0.0],
                       "clara_lon": [0.0, 0.1, 0.0, 0.0], "vza": [10.0, 70.0, 10.0, 10.0],
                       "earth_view": [True, True, True, True]})
    f = samples.add_flags(df, QC)
    assert list(f.qc) == [True, False, False, False]  # VZA too large, daytime, sun-viewing value
    assert list(f.overpass) == [0, -1, -1, -1]


def test_solar_zenith_reference_values():
    # equinox: sun near overhead at the equator at local noon, near nadir at midnight
    t = pd.to_datetime(["2021-03-20 12:00", "2021-03-20 00:00", "2021-06-21 12:00", "2021-12-21 12:00"])
    sza = solar.solar_zenith(t, [0, 0, 23.44, -23.44], [0, 0, 0, 0])
    assert sza[0] < 2.5 and sza[1] > 177 and sza[2] < 1.5 and sza[3] < 1.5
    # local solar time at 90 E is 6 h ahead of Greenwich (up to the equation of time, < 17 min)
    lst = solar.local_solar_time(pd.to_datetime(["2021-07-01 00:00"]), [90.0])
    assert abs(lst[0] - 6.0) < 0.3


def test_night_weights_trivial_cases():
    w = era5_night.night_weights(pd.Period("2021-12", "M"), np.array([-89.5, 0.5, 89.5]), np.array([0.5]), 100.0)
    assert np.all(w[:, 0, 0] == 0)  # southern summer: the south pole never has night
    assert np.all(w[:, 2, 0] == 1)  # northern winter: the north pole is always night
    assert 0.3 < w[:, 1, 0].mean() < 0.5  # equator: ~10 of 24 h beyond SZA 100


def test_fallback_uses_the_darkest_hours():
    w, mean_cos = era5_night.sun_geometry(pd.Period("2021-12", "M"), np.array([-89.5, 0.5]), np.array([0.5]), 100.0)
    wf, fb = era5_night.fallback_weights(w, mean_cos, 3)
    assert list(fb[:, 0]) == [True, False]
    assert np.allclose(wf[:, 0, 0].sum(), 1.0) and (wf[:, 0, 0] > 0).sum() == 3
    assert set(np.flatnonzero(wf[:, 0, 0])) == set(np.argsort(mean_cos[:, 0, 0])[:3])
    assert np.array_equal(wf[:, 1, 0], w[:, 1, 0])  # cells with a night keep their weights


def test_cap_smoothing_constant_nan_and_wrap():
    lat, lon = np.arange(-89.5, 90), np.arange(-179.5, 180)
    f = np.full((180, 360), 7.0)
    f[:3] = np.nan
    s = era5_night.cap_smooth(f, lat, lon, 500.0)
    assert np.allclose(s[np.isfinite(s)], 7.0)
    g = np.zeros((180, 360))
    g[90, 0] = 1.0
    s = era5_night.cap_smooth(g, lat, lon, 400.0)
    cols = np.unique(np.argwhere(s > 1e-9)[:, 1])  # above FFT round-off
    assert 359 in cols and 0 in cols and 180 not in cols  # spreads across the dateline, not beyond the cap
    ds = xr.Dataset({"v": (("lat", "lon"), f)}, coords={"lat": lat, "lon": lon})
    sm = era5_night.smoothed(ds, "v", [0.0, 500.0])
    assert np.isnan(sm.v_r500.values[:3]).all()  # NaN cells (polar day) are never filled


def test_post_gap_indicator_and_mid_month():
    t = pd.to_datetime(["2020-01-01 00:00", "2021-08-01 00:00", "2021-08-31 23:00", "2021-09-01 00:00", "2022-01-01 00:00",
                        "2022-06-01 00:00"])
    assert list(tables.post_gap_indicator(t, "2021-09")) == [0, 0, 0, 1, 1, 1]  # gap months count as post-gap
    mid = tables.mid_month(pd.to_datetime(["2021-02-01", "2020-02-01", "2021-01-01"]))
    assert list(pd.DatetimeIndex(mid)) == list(pd.to_datetime(["2021-02-15 00:00", "2020-02-15 12:00", "2021-01-16 12:00"]))
