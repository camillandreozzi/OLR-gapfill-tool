"""Read raw CLARA (NorSat-1) OLR and its geolocation, and put every OLR sample at its footprint.

Expected input (paths.clara_dir), as distributed by the instrument team; archives, extracted files or both:

    CLARA_OLR*.tar.gz            IDL .save files, one per month, CLARA_OLR_<yyyy>_<m>.save. Each holds a struct
                                 OLR_ALL with JDAY (Julian date, UTC) and OLR (radiance, W m-2 sr-1) at ~30 s
                                 cadence. The OLR channel also sees the Sun: values of ~1e4-9e4 are sun-viewing
                                 periods, not Earth OLR (removed by the qc.olr_max limit).
    CLARA_lat_lon_<yyyy>.tar.gz  CSV files, one per day, CLARA_lon_lat_alt-<yyyy>-<m>-<d>_V50_1_4[_test].csv, at
                                 1 min cadence: satellite position and CLARA footprint as "(lat, lon, alt_m)"
                                 tuples (WGS84 geodetic). The footprint is "(0, 0, 0)" when CLARA looks away from
                                 the Earth and empty when its line of sight passes above the limb.

Archives are extracted into <work_dir>/extracted/. Loose .save / .csv files anywhere under clara_dir (outside
folders named "extracted") are read as well and take precedence over a month or day found in an archive; a
month in a later archive (sorted by name, so CLARA_OLR_<yyyy> after CLARA_OLR) replaces an earlier one; for a
day stored both as a final and a "_test" geolocation file, the final one is used.

The footprint is recomputed as the near intersection of the satellite -> footprint line with a 6371 km sphere
(older releases stored the far intersection), which also gives the viewing zenith angle (vza).
"""

import os
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

JULIAN_UNIX_EPOCH = 2440587.5  # Julian date of 1970-01-01T00:00Z

# The file footprints lie exactly on this sphere (|ECEF| = 6371.000 km for every row).
FOOTPRINT_SPHERE_R = 6_371_000.0  # m
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)  # WGS84 (lon, lat, h) -> ECEF
_FROM_ECEF = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

OLR_FILE_RE = re.compile(r"^CLARA_OLR_(\d{4})_(\d{1,2})\.save$")
GEO_FILE_RE = re.compile(r"^CLARA_lon_lat_alt-(\d{4})-(\d{1,2})-(\d{1,2})_V50_1_4(?:_test)?\.csv$")
TUPLE_RE = r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)"
GEO_COLUMNS = ("z_angle", "sat_lat", "sat_lon", "sat_alt", "clara_lat", "clara_lon", "vza")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _extract(archive, extract_dir):
    """Extract a tar.gz archive into extract_dir/<archive stem>/ and return that folder.

    The extraction is reused until the archive is replaced by a newer one.
    """
    out = Path(extract_dir) / archive.name.removesuffix(".tar.gz")
    if out.exists() and archive.stat().st_mtime > out.stat().st_mtime:
        print(f"{archive.name} is newer than its extraction; extracting again")
        shutil.rmtree(out)
    if not out.exists():
        tmp = out.with_name(out.name + ".part")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        tmp.rename(out)
        os.utime(out)  # stamp the extraction time, compared with the archive's on the next call
    return out


def _loose(clara_dir, suffix):
    return sorted(p for p in Path(clara_dir).rglob(f"*{suffix}") if "extracted" not in p.relative_to(clara_dir).parts)


def source_files(clara_dir):
    """Every input file under clara_dir (archives and loose files), for the stage manifest."""
    clara_dir = Path(clara_dir)
    return (sorted(clara_dir.glob("CLARA_OLR*.tar.gz")) + sorted(clara_dir.glob("CLARA_lat_lon_*.tar.gz"))
            + _loose(clara_dir, ".save") + _loose(clara_dir, ".csv"))


def _in_years(year, years):
    return years is None or year in years


def olr_files(clara_dir, extract_dir, years=None):
    """{(year, month): .save path}, archives first, loose files last (they take precedence)."""
    clara_dir = Path(clara_dir)
    paths = {}
    # "CLARA_OLR.tar.gz" sorts before "CLARA_OLR_<yyyy>.tar.gz", so a month in a yearly archive wins.
    candidates = [p for a in sorted(clara_dir.glob("CLARA_OLR*.tar.gz")) for p in sorted(_extract(a, extract_dir).rglob("*.save"))]
    for path in candidates + _loose(clara_dir, ".save"):
        m = OLR_FILE_RE.search(path.name)
        if m and _in_years(int(m.group(1)), years):
            paths[int(m.group(1)), int(m.group(2))] = path
    return paths


def geo_files(clara_dir, extract_dir, years=None):
    """{date: .csv path}: a final file over a "_test" one, then a loose file over an archived one."""
    clara_dir = Path(clara_dir)
    candidates = [(1, p) for a in sorted(clara_dir.glob("CLARA_lat_lon_*.tar.gz"))
                  for p in sorted(_extract(a, extract_dir).rglob("*.csv"))]
    candidates += [(0, p) for p in _loose(clara_dir, ".csv")]
    best = {}
    for rank, path in candidates:
        m = GEO_FILE_RE.search(path.name)
        if not m or not _in_years(int(m.group(1)), years):
            continue
        date = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        key = ("_test" in path.name, rank)
        if date not in best or key < best[date][0]:
            best[date] = (key, path)
    return {d: p for d, (_, p) in best.items()}


# ---------------------------------------------------------------------------
# OLR
# ---------------------------------------------------------------------------

def read_clara_olr(clara_dir, extract_dir, years=None):
    """The CLARA OLR time series as a DataFrame [time, olr], every sample as stored (sun-viewing ones too)."""
    from scipy.io import readsav

    paths = olr_files(clara_dir, extract_dir, years)
    if not paths:
        raise FileNotFoundError(f"no CLARA_OLR_<yyyy>_<m>.save files (or CLARA_OLR*.tar.gz archives) in {clara_dir}")
    frames = []
    for key in sorted(paths):
        rec = readsav(str(paths[key]))["olr_all"][0]
        # IDL save files are big-endian; convert to native floats for pandas.
        jday = np.asarray(rec["JDAY"], dtype=np.float64)
        olr = np.asarray(rec["OLR"], dtype=np.float64)
        frames.append(pd.DataFrame({
            "time": pd.to_datetime(jday - JULIAN_UNIX_EPOCH, unit="D").round("ms"),
            "olr": olr,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geolocation
# ---------------------------------------------------------------------------

def _ecef(lat, lon, alt):
    return np.stack(_TO_ECEF.transform(lon, lat, alt), axis=-1)


def _latlon_to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def _xyz_to_latlon(x, y, z):
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def near_footprint(sat_lat, sat_lon, sat_alt, fp_lat, fp_lon, fp_alt):
    """Return (lat, lon, vza) of the near intersection of the satellite -> file-footprint line with the
    6371 km sphere, i.e. the point CLARA looks at. For current files this is the file footprint itself;
    older releases stored the far intersection of the same line, which this maps back to the near one.

    vza is the viewing zenith angle at the footprint (deg; 0 = satellite overhead, 90 = grazing the limb),
    measured from the WGS84 surface normal.
    """
    s = _ecef(sat_lat, sat_lon, sat_alt)
    d = _ecef(fp_lat, fp_lon, fp_alt) - s
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    b = np.sum(s * d, axis=-1)
    disc = b * b - (np.sum(s * s, axis=-1) - FOOTPRINT_SPHERE_R ** 2)
    f = s + (-b - np.sqrt(np.maximum(disc, 0)))[:, None] * d
    lon, lat, _ = _FROM_ECEF.transform(f[:, 0], f[:, 1], f[:, 2])
    up = np.stack(_latlon_to_xyz(lat, lon), axis=-1)
    vza = np.degrees(np.arccos(np.clip(np.sum(up * -d, axis=-1), -1, 1)))
    return lat, lon, vza


def read_geo_csv(path, date):
    """One day of geolocation: time, z_angle, sat_lat, sat_lon, sat_alt, clara_lat, clara_lon, vza, earth_view."""
    df = pd.read_csv(path)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]
    sat = df["satellite_lat_long"].str.extract(TUPLE_RE).astype(float).to_numpy()
    fov = df["clara_lat_long"].str.extract(TUPLE_RE).astype(float).to_numpy()
    # "(0, 0, 0)" = looking away from the Earth, empty (NaN) = line of sight above the limb.
    earth_view = np.isfinite(fov).all(axis=1) & ~(fov == 0).all(axis=1)
    lat, lon, vza = np.full((3, len(df)), np.nan)
    if earth_view.any():
        lat[earth_view], lon[earth_view], vza[earth_view] = near_footprint(*sat[earth_view].T, *fov[earth_view].T)
    return pd.DataFrame({
        "time": date + pd.to_timedelta(df["datetime"]),
        "z_angle": df["z_vector_satellite_position_angles"].astype(float),
        "sat_lat": sat[:, 0], "sat_lon": sat[:, 1], "sat_alt": sat[:, 2],
        "clara_lat": lat, "clara_lon": lon, "vza": vza,
        "earth_view": earth_view,
    })


def _geo_empty():
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in GEO_COLUMNS})
    df.insert(0, "time", pd.Series(dtype="datetime64[ns]"))
    df["earth_view"] = pd.Series(dtype=bool)
    return df


def read_clara_geolocation(clara_dir, extract_dir, years=None):
    """The 1-minute CLARA geolocation of every day found (see read_geo_csv for the columns)."""
    paths = geo_files(clara_dir, extract_dir, years)
    frames = [read_geo_csv(paths[date], date) for date in sorted(paths)]
    if not frames:  # e.g. OLR years without geolocation files yet
        return _geo_empty()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time", kind="stable").drop_duplicates("time").reset_index(drop=True)


def geolocate(olr, geo, max_gap_s):
    """Interpolate the 1-minute geolocation onto the OLR sample times.

    Positions are interpolated on the unit sphere so the dateline and poles are handled correctly. A sample
    is only geolocated when both neighbouring geolocation rows are at most max_gap_s apart; the footprint is
    only filled when both neighbours are Earth viewing.
    """
    if len(geo) < 2:  # nothing to interpolate between, so no sample is geolocated
        out = olr.copy()
        for col in GEO_COLUMNS:
            out[col] = np.nan
        out["earth_view"] = False
        return out
    tg = geo["time"].to_numpy("datetime64[ns]").astype(np.int64)
    to = olr["time"].to_numpy("datetime64[ns]").astype(np.int64)
    right = np.searchsorted(tg, to).clip(1, len(tg) - 1)
    left = right - 1
    dt = tg[right] - tg[left]
    ok = (to >= tg[0]) & (to <= tg[-1]) & (dt <= int(round(max_gap_s * 1e9)))
    w = np.where(dt > 0, (to - tg[left]) / np.maximum(dt, 1), 0.0)

    def lerp(col):
        a, b = geo[col].to_numpy()[left], geo[col].to_numpy()[right]
        return np.where(ok, a + w * (b - a), np.nan)

    def lerp_latlon(prefix):
        xyz_l = _latlon_to_xyz(geo[f"{prefix}_lat"].to_numpy()[left], geo[f"{prefix}_lon"].to_numpy()[left])
        xyz_r = _latlon_to_xyz(geo[f"{prefix}_lat"].to_numpy()[right], geo[f"{prefix}_lon"].to_numpy()[right])
        lat, lon = _xyz_to_latlon(*(a + w * (b - a) for a, b in zip(xyz_l, xyz_r)))
        return np.where(ok, lat, np.nan), np.where(ok, lon, np.nan)

    ev = geo["earth_view"].to_numpy()
    out = olr.copy()
    out["z_angle"] = lerp("z_angle")
    out["sat_lat"], out["sat_lon"] = lerp_latlon("sat")
    out["sat_alt"] = lerp("sat_alt")
    out["clara_lat"], out["clara_lon"] = lerp_latlon("clara")
    out["vza"] = lerp("vza")
    out["earth_view"] = ok & ev[left] & ev[right]
    return out


def read_clara(clara_dir, extract_dir, max_gap_s, years=None):
    """Every CLARA OLR sample with its interpolated geometry (columns of geolocate)."""
    return geolocate(read_clara_olr(clara_dir, extract_dir, years),
                     read_clara_geolocation(clara_dir, extract_dir, years), max_gap_s)
