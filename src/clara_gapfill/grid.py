"""Regular global lat-lon grids: cell edges and centres, cell lookup, cell areas and conservative regridding.

Conventions: latitudes ascending from -90, longitudes in [-180, 180), cells of res x res deg starting at
(-90, -180); res divides 180, so the grid has 180/res x 360/res cells.
"""

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0


def grid_edges(res):
    """Cell edges (lat, lon) of the res x res deg grid starting at (-90, -180)."""
    res = int(res)
    return (np.minimum(np.arange(-90, 90 + res, res), 90).astype(float),
            np.minimum(np.arange(-180, 180 + res, res), 180).astype(float))


def grid_centres(res):
    """Cell centres (lat, lon) of grid_edges(res)."""
    return tuple((e[:-1] + e[1:]) / 2 for e in grid_edges(res))


def cell_index(lat, lon, res):
    """Row and column of the res deg cells holding each (lat, lon); lat = 90 and lon = 180 fall in the last
    row / column."""
    lat_c, lon_c = grid_centres(res)
    i = np.clip(np.floor((np.asarray(lat) + 90) / res).astype(int), 0, len(lat_c) - 1)
    j = np.clip(np.floor((np.asarray(lon) + 180) / res).astype(int), 0, len(lon_c) - 1)
    return i, j


def cell_area_km2(lat_c, res):
    """Area (km2) of a res x res deg cell centred at lat_c (exact on the sphere)."""
    lat_edges, _ = grid_edges(res)
    i = np.clip(np.floor((np.asarray(lat_c) + 90) / res).astype(int), 0, len(lat_edges) - 2)
    s, n = np.radians(lat_edges[i]), np.radians(lat_edges[i + 1])
    return EARTH_RADIUS_KM ** 2 * (np.sin(n) - np.sin(s)) * np.radians(res)


def standardise_grid(ds):
    """Rename coords to (time, lat, lon), lat ascending, lon in [-180, 180)."""
    ds = ds.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
                    if k in ds.dims or k in ds.coords})
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    return ds.sortby(["lat", "lon"])


def era5_to_1deg(ds):
    """Area-average ERA5 from its native 0.25 deg point grid onto 1 deg cells.

    ERA5 points sit on cell edges (lat -90..90, lon -180..179.75), so each 1 deg box spans 5 points per axis;
    the edge points get half weight (trapezoid rule) and latitudes are weighted by cos(lat). Longitudes wrap
    at the dateline.
    """
    lat, lon = ds.lat.values, ds.lon.values
    if len(lat) != 721 or len(lon) != 1440 or lat[0] != -90 or lon[0] != -180:
        raise ValueError("expected the native ERA5 0.25 deg grid (721 x 1440); download ERA5 without a 'grid' "
                         "option (clara-gapfill download-era5 does)")
    trap = np.array([0.5, 1, 1, 1, 0.5])
    w_lat = np.zeros((180, 721))
    w_lon = np.zeros((360, 1440))
    for i in range(180):
        w = trap * np.cos(np.radians(lat[4 * i:4 * i + 5]))
        w_lat[i, 4 * i:4 * i + 5] = w / w.sum()
    for k in range(360):
        w_lon[k, np.arange(4 * k, 4 * k + 5) % 1440] += trap / trap.sum()

    out = xr.Dataset(coords={"time": ds.time, "lat": np.arange(-89.5, 90), "lon": np.arange(-179.5, 180)})
    for v in ds.data_vars:
        out[v] = (("time", "lat", "lon"), (w_lat @ ds[v].values.astype(np.float64)) @ w_lon.T)
        out[v].attrs = ds[v].attrs
    return out


def coarsen(ds, res):
    """Area-weighted, NaN-aware average of a 1 deg (time, lat, lon) Dataset onto the res x res deg cells."""
    if res == 1:
        return ds
    w = np.cos(np.radians(ds.lat)) * xr.ones_like(ds.lon)
    lat, lon = grid_centres(res)
    out = xr.Dataset()
    for v in ds.data_vars:
        wv = w.where(ds[v].notnull())
        num = (ds[v] * wv).coarsen(lat=res, lon=res, boundary="pad").sum()
        den = wv.coarsen(lat=res, lon=res, boundary="pad").sum()
        out[v] = (num / den).assign_coords(lat=lat, lon=lon)
        out[v].attrs = ds[v].attrs
    return out
