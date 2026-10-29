"""Solar geometry, QC flags and overpass ids of the geolocated CLARA samples.

add_flags() adds:
    sza        solar zenith angle at the footprint (deg)
    lst        local solar time at the footprint (h)
    in_range   olr within [qc.olr_min (raised per qc.olr_min_from), qc.olr_max]
    usable     earth_view & in_range
    night      sza > qc.night_sza_min (CLARA measures OLR on the night side only)
    vza_ok     vza <= qc.vza_max
    qc         usable & night & vza_ok: the observations the model uses
    overpass   id of the overpass a qc sample belongs to (-1 otherwise); a new overpass starts after a gap of
               more than qc.overpass_gap_min minutes between consecutive qc samples
"""

import numpy as np
import pandas as pd

from . import solar


def olr_min(time, qc):
    """Lower plausible limit (W m-2 sr-1) for each sample time: qc.olr_min, raised per qc.olr_min_from."""
    year = pd.Series(time).dt.year.to_numpy()
    lo = np.full(year.shape, float(qc["olr_min"]))
    for first, limit in sorted(qc["olr_min_from"].items()):
        lo[year >= int(first)] = limit
    return lo


def in_range(olr, time, qc):
    return (np.asarray(olr) >= olr_min(time, qc)) & (np.asarray(olr) <= qc["olr_max"])


def overpass_ids(time, gap_min):
    """Overpass id per (sorted) sample time: a new overpass starts after a gap longer than gap_min."""
    t = pd.to_datetime(time).to_numpy("datetime64[ns]").astype(np.int64)
    new = np.r_[True, np.diff(t) > gap_min * 60e9]
    return np.cumsum(new) - 1


def add_flags(df, qc):
    """Add sza, lst and the QC flags of the module docstring to geolocated samples (a copy)."""
    df = df.copy()
    has_fp = df.clara_lat.notna().to_numpy()
    df["sza"] = np.nan
    df["lst"] = np.nan
    df.loc[has_fp, "sza"] = solar.solar_zenith(df.time[has_fp], df.clara_lat[has_fp], df.clara_lon[has_fp])
    df.loc[has_fp, "lst"] = solar.local_solar_time(df.time[has_fp], df.clara_lon[has_fp])
    df["in_range"] = in_range(df.olr, df.time, qc)
    df["usable"] = df.earth_view & df.in_range
    df["night"] = df.sza > qc["night_sza_min"]
    df["vza_ok"] = df.vza <= qc["vza_max"]
    df["qc"] = df.usable & df.night & df.vza_ok
    df["overpass"] = -1
    sel = df.qc.to_numpy()
    df.loc[sel, "overpass"] = overpass_ids(df.time[sel], qc["overpass_gap_min"])
    return df


def qc_counts(s, qc):
    """Samples left after each QC stage."""
    stages = [("all samples", np.ones(len(s), bool)),
              ("geolocated", s.sat_lat.notna().to_numpy()),
              ("Earth viewing", s.earth_view.to_numpy()),
              ("Earth viewing & in plausible range", s.usable.to_numpy()),
              (f"+ night side (SZA > {qc['night_sza_min']:g} deg)", (s.usable & s.night).to_numpy()),
              (f"+ VZA <= {qc['vza_max']:g} deg (model observations)", s.qc.to_numpy())]
    usable = max(int(s.usable.sum()), 1)
    return pd.DataFrame([{"stage": k, "samples": int(v.sum()), "fraction_of_usable": v.sum() / usable}
                         for k, v in stages])
