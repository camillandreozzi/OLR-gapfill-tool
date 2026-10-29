"""ERA5 monthly night-time OLR on 1 deg cells, built from the ERA5 monthly means by hour of day.

CLARA measures OLR on the night side only, while an ERA5 monthly mean covers 24 h. Per 1 deg cell and month:

    OLR_h       = -ttr_h / 3600                                   W m-2, positive upward
    w_h         = fraction of the month's days on which the solar zenith angle at the cell centre, at the
                  middle of hour h, exceeds era5.night_sza_min (the CLARA night definition)
    olr_night   = sum_h w_h OLR_h / sum_h w_h
                  where no hour is ever night (polar day) the night-time value does not exist; so that the
                  prediction grid has no gaps, these cells use the mean of the era5.fallback_hours hours with
                  the lowest sun (around local solar midnight) and are flagged night_fallback = 1. CLARA never
                  observes them, so predictions there are extrapolation.
    olr_24h     = mean_h OLR_h
    night_frac  = sum_h w_h / 24                                  fraction of the month that is night (0 = polar day)

The 0.25 deg points are first area-averaged onto 1 deg cells per hour (grid.era5_to_1deg), so the night weights
are evaluated at 1 deg resolution.

cap_smooth() averages a gridded field over spherical caps (area weighted, NaN aware). It uses ERA5 only and
degrades its effective resolution towards CLARA's wide footprint.
"""

import numpy as np
import pandas as pd
import xarray as xr

from . import era5, grid, solar

EARTH_RADIUS_KM = 6371.0


def night_weights(month, lat, lon, sza_min):
    """(24, nlat, nlon) fraction of the days of month on which hour h (ending at h+1 UTC) is night."""
    return sun_geometry(month, lat, lon, sza_min)[0]


def sun_geometry(month, lat, lon, sza_min):
    """Night weights (see night_weights) and the monthly-mean cos(SZA) of each hour, both (24, nlat, nlon)."""
    start = month.to_timestamp()
    days = pd.date_range(start, periods=month.days_in_month, freq="D")
    phi = np.radians(np.asarray(lat))[:, None]
    lon = np.asarray(lon)[None, :]
    cos_min = np.cos(np.radians(sza_min))
    w = np.zeros((24, phi.shape[0], lon.shape[1]))
    mean_cos = np.zeros_like(w)
    for day in days:
        mid = day + pd.to_timedelta(np.arange(24) + 0.5, unit="h")  # middle of the hours ending 01..24 UTC
        decl, eot = solar.declination_eot(mid)
        utc_min = (np.arange(24) + 0.5) * 60
        for h in range(24):
            ha = np.radians(((utc_min[h] + eot[h] + 4 * lon) % 1440) / 4 - 180)
            d = np.radians(decl[h])
            cos_sza = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(ha)
            w[h] += cos_sza < cos_min  # sza > sza_min
            mean_cos[h] += cos_sza
    return w / len(days), mean_cos / len(days)


def fallback_weights(w, mean_cos, k):
    """Weights with polar-day cells (no night hour) replaced by 1/k on their k darkest hours.
    Returns (weights, fallback mask)."""
    fb = w.sum(0) == 0
    if not fb.any():
        return w, fb
    darkest = np.argsort(mean_cos, axis=0)[:k]  # smallest cos(SZA) = sun lowest
    wf = np.zeros_like(w)
    np.put_along_axis(wf, darkest, 1.0 / k, axis=0)
    return np.where(fb[None], wf, w), fb


def _area_mean(da):
    w = np.cos(np.radians(da.lat)) * xr.ones_like(da)
    return float(da.weighted(w.where(da.notnull(), 0).fillna(0)).mean())


def build(era5_dir, months, sza_min, k):
    """Night-time product of every month of `months` (Periods) found in era5_dir.
    Returns (Dataset (time, lat, lon) on 1 deg cells, per-month check table)."""
    wanted = set(months)
    out, checks = {}, {}
    for path in era5.files(era5_dir):
        for month, hourly in era5.month_blocks(path, wanted):  # one month at a time (~200 MB at 0.25 deg)
            if month in out:
                print(f"warning: {month} is in more than one ERA5 file; using {path.name}")
            one = grid.era5_to_1deg(hourly)  # (24, 180, 360) area means on 1 deg cells
            w, mean_cos = sun_geometry(month, one.lat.values, one.lon.values, sza_min)
            olr = one.olr.values
            night_frac = w.sum(0) / 24
            w, fallback = fallback_weights(w, mean_cos, k)
            night = (w * olr).sum(0) / w.sum(0)
            ds = xr.Dataset({"olr_night": (("lat", "lon"), night),
                             "olr_24h": (("lat", "lon"), olr.mean(0)),
                             "night_frac": (("lat", "lon"), night_frac),
                             "night_fallback": (("lat", "lon"), fallback.astype(np.int8))},
                            coords={"lat": one.lat, "lon": one.lon}).expand_dims(time=[month.to_timestamp()])
            out[month] = ds
            checks[month] = {"month": str(month), "global_mean_olr_24h": _area_mean(ds.olr_24h),
                             "global_mean_olr_night": _area_mean(ds.olr_night),
                             "polar_day_cells_fallback": int(fallback.sum())}
            print(f"  {month}: night-time mean built ({checks[month]['polar_day_cells_fallback']} polar-day cells "
                  f"use the {k} darkest hours)", flush=True)
    if not out:
        raise FileNotFoundError(f"no ERA5 month of {months[0]}..{months[-1]} in {era5_dir}")
    ds = xr.concat([out[m] for m in sorted(out)], dim="time")
    checks = [checks[m] for m in sorted(checks)]
    ds.olr_night.attrs = {"units": "W m-2", "long_name": f"ERA5 monthly mean night-time OLR (SZA > {sza_min:g} deg); "
                          f"polar-day cells: mean of the {k} darkest hours (night_fallback = 1)"}
    ds.night_fallback.attrs = {"units": "1", "long_name": "1 where the cell has no night that month (polar day)"}
    ds.olr_24h.attrs = {"units": "W m-2", "long_name": "ERA5 monthly mean OLR (24 h)"}
    ds.night_frac.attrs = {"units": "1", "long_name": "fraction of the month that is night at the cell centre"}
    ds.attrs = {"source": "ERA5 monthly averaged reanalysis by hour of day, ttr; built by clara-gapfill",
                "night_sza_min_deg": sza_min, "fallback_hours": k}
    return ds, pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# Spherical-cap smoothing
# ---------------------------------------------------------------------------

def cap_kernel(lat, lon, radius_km):
    """rfft along longitude of the cap weights K[i, j, dlon]: area weight of source row j inside the cap
    of radius radius_km around (lat_i, 0). Rows are exact: the cap depends only on (lat_i, lat_j, dlon)."""
    phi = np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - lon[0])
    cosd = (np.sin(phi)[:, None, None] * np.sin(phi)[None, :, None]
            + np.cos(phi)[:, None, None] * np.cos(phi)[None, :, None] * np.cos(dl)[None, None, :])
    inside = cosd >= np.cos(radius_km / EARTH_RADIUS_KM)
    k = inside * np.cos(phi)[None, :, None]  # cell area is proportional to cos(lat) on a regular grid
    return np.fft.rfft(k, axis=-1)


def cap_smooth(values, lat, lon, radius_km):
    """Area-weighted, NaN-aware mean of values (..., nlat, nlon) over spherical caps of radius_km around
    each cell centre of a regular global grid. radius_km <= 0 returns the input unchanged."""
    values = np.asarray(values, float)
    if radius_km <= 0:
        return values.copy()
    kh = cap_kernel(lat, lon, radius_km)  # (i, j, f)
    n = values.shape[-1]
    valid = np.isfinite(values)
    lead = values.shape[:-2]
    f = np.fft.rfft(np.where(valid, values, 0.0).reshape(-1, *values.shape[-2:]), axis=-1)  # (b, j, f)
    v = np.fft.rfft(valid.reshape(-1, *values.shape[-2:]).astype(float), axis=-1)
    kt = np.transpose(kh, (2, 1, 0))  # (f, j, i)
    num = np.fft.irfft(np.transpose(np.matmul(np.transpose(f, (2, 0, 1)), kt), (1, 2, 0)), n=n, axis=-1)
    den = np.fft.irfft(np.transpose(np.matmul(np.transpose(v, (2, 0, 1)), kt), (1, 2, 0)), n=n, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9 * den.max(), num / den, np.nan)
    return out.reshape(*lead, *values.shape[-2:])


def smoothed(ds, var, radii_km):
    """Dataset with var smoothed at each radius, as f'{var}_r{int(R)}' (r0 = unsmoothed).

    Cells that are NaN in var stay NaN; smoothing never fills them.
    """
    out = xr.Dataset(coords=ds.coords)
    for r in radii_km:
        sm = cap_smooth(ds[var].values, ds.lat.values, ds.lon.values, r)
        out[f"{var}_r{int(r)}"] = (ds[var].dims, np.where(np.isfinite(ds[var].values), sm, np.nan))
        out[f"{var}_r{int(r)}"].attrs = {**ds[var].attrs, "cap_radius_km": r}
    return out
