"""CLARA geolocation files: parsing, footprint geometry, file precedence and interpolation onto OLR samples."""

import numpy as np
import pandas as pd
import pytest

from clara_gapfill import clara

HEADER = "# datetime,z_vector_satellite_position_angles,satellite_lat_long,clara_lat_long\n"


def write_geo(path, rows):
    path.write_text(HEADER + "".join(f'{t},{z},"{s}","{f}"\n' for t, z, s, f in rows))


def test_geo_csv_parsing_and_earth_view(tmp_path):
    p = tmp_path / "CLARA_lon_lat_alt-2021-3-5_V50_1_4.csv"
    write_geo(p, [("00:00:00", 10.0, "(10.0, 20.0, 590000.0)", "(10.5, 20.0, 0.0)"),
                  ("00:01:00", 120.0, "(10.5, 20.0, 590000.0)", "(0, 0, 0)"),
                  ("00:02:00", 80.0, "(11.0, 20.0, 590000.0)", "")])
    g = clara.read_geo_csv(p, pd.Timestamp("2021-03-05"))
    assert list(g.earth_view) == [True, False, False]
    assert g.time.iloc[1] == pd.Timestamp("2021-03-05 00:01")
    # this footprint is on the WGS84 ellipsoid, not on the 6371 km sphere of the files, hence the tolerance
    assert g.clara_lat.iloc[0] == pytest.approx(10.5, abs=0.02) and np.isnan(g.clara_lat.iloc[1])
    assert 0 < g.vza.iloc[0] < 10  # a footprint half a degree off nadir is seen almost from overhead


def test_near_footprint_maps_the_far_intersection_back():
    sat = (0.0, 0.0, 600_000.0)
    near = clara.near_footprint(*[np.array([v]) for v in (*sat, 0.0, 2.0, 0.0)])
    s = clara._ecef(*[np.array([v]) for v in sat])
    d = clara._ecef(np.array([0.0]), np.array([2.0]), np.array([0.0])) - s
    d /= np.linalg.norm(d)
    b = (s * d).sum()
    far = s + (-b + np.sqrt(b * b - ((s * s).sum() - clara.FOOTPRINT_SPHERE_R ** 2))) * d
    lon, lat, alt = clara._FROM_ECEF.transform(far[:, 0], far[:, 1], far[:, 2])
    back = clara.near_footprint(*[np.array([v]) for v in sat], lat, lon, alt)
    assert back[0][0] == pytest.approx(near[0][0], abs=1e-6) and back[1][0] == pytest.approx(near[1][0], abs=1e-6)


def test_final_file_wins_over_test_file(tmp_path):
    row = [("00:00:00", 10.0, "(10.0, 20.0, 590000.0)", "(10.5, 20.0, 0.0)")]
    write_geo(tmp_path / "CLARA_lon_lat_alt-2021-3-5_V50_1_4_test.csv", row)
    write_geo(tmp_path / "CLARA_lon_lat_alt-2021-3-5_V50_1_4.csv", row)
    files = clara.geo_files(tmp_path, tmp_path / "extracted")
    assert [p.name for p in files.values()] == ["CLARA_lon_lat_alt-2021-3-5_V50_1_4.csv"]


def test_geolocate_interpolates_and_respects_gaps():
    geo = pd.DataFrame({"time": pd.to_datetime(["2021-01-01 00:00", "2021-01-01 00:01", "2021-01-01 00:05"]),
                        "z_angle": [0.0, 10.0, 20.0], "sat_lat": [0.0, 1.0, 5.0], "sat_lon": [179.5, -179.5, -175.0],
                        "sat_alt": [6e5, 6e5, 6e5], "clara_lat": [0.0, 1.0, 5.0], "clara_lon": [179.5, -179.5, -175.0],
                        "vza": [5.0, 5.0, 5.0], "earth_view": [True, True, True]})
    olr = pd.DataFrame({"time": pd.to_datetime(["2021-01-01 00:00:30", "2021-01-01 00:03:00"]), "olr": [200.0, 210.0]})
    out = clara.geolocate(olr, geo, 65.0)
    assert abs(out.clara_lon.iloc[0]) == pytest.approx(180.0, abs=1e-3)  # across the dateline, not via 0
    assert out.z_angle.iloc[0] == pytest.approx(5.0) and out.earth_view.iloc[0]
    assert np.isnan(out.clara_lat.iloc[1]) and not out.earth_view.iloc[1]  # 4 min between geolocation rows
