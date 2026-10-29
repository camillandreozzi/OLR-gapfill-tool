"""ERA5 top net thermal radiation (ttr = -OLR) "monthly averaged reanalysis by hour of day": download and read.

The product holds, for each month, 24 fields: the monthly mean of the ttr accumulation over the hour ending at
each UTC hour (J m-2, negative upward), on the native 0.25 deg grid. One file per year is downloaded to
<paths.era5_dir>/era5_ttr_monthly_by_hour_<yyyy>.nc; any NetCDF files of this product in era5_dir are read.

Downloading needs a Copernicus Climate Data Store account with the ERA5 licence accepted, the cdsapi package
(pip install "clara-olr-gapfill[era5]") and credentials in ~/.cdsapirc:

    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import grid

DATASET = "reanalysis-era5-single-levels-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis_by_hour_of_day"
HOURS = [f"{h:02d}:00" for h in range(24)]
ERA5_LAG = dt.timedelta(days=6)  # a month is requested once it ended at least this long ago
CDSAPIRC = Path.home() / ".cdsapirc"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def published(period, today=None):
    """True when ERA5 has the whole month: it ended at least ERA5_LAG ago."""
    end = (period + 1).to_timestamp().date()
    return end + ERA5_LAG <= (today or dt.date.today())


def requests(months, era5_dir, today=None):
    """[(target path, request dict)], one per year, for the published months of `months` (Periods)."""
    out = []
    by_year = {}
    for p in months:
        if published(p, today):
            by_year.setdefault(p.year, []).append(f"{p.month:02d}")
    for year, mm in sorted(by_year.items()):
        out.append((Path(era5_dir) / f"era5_ttr_monthly_by_hour_{year}.nc",
                    {"product_type": [PRODUCT_TYPE], "variable": ["top_net_thermal_radiation"], "year": [str(year)],
                     "month": mm, "time": HOURS, "data_format": "netcdf", "download_format": "unarchived"}))
    return out


def _n_times(path):
    if not path.exists():
        return 0
    with xr.open_dataset(path) as ds:
        return ds.sizes.get("valid_time", ds.sizes.get("time", 0))


def download(months, era5_dir, dry_run=False):
    """Download the by-hour product for `months`; complete yearly files are skipped, so re-running resumes."""
    reqs = requests(months, era5_dir)
    skipped = [p for p in months if not published(p)]
    if skipped:
        print(f"not yet published by ERA5, skipped: {', '.join(str(p) for p in skipped)}")
    if dry_run:
        for target, req in reqs:
            print(f"{DATASET} -> {target}\n  {req}")
        return reqs
    try:
        import cdsapi
    except ImportError as err:
        raise SystemExit('download-era5 needs cdsapi: pip install "clara-olr-gapfill[era5]"') from err
    if not CDSAPIRC.exists():
        raise SystemExit(f"Missing {CDSAPIRC}. Create it with your CDS url and key "
                         "(see https://cds.climate.copernicus.eu/how-to-api).")
    client = cdsapi.Client()
    for target, req in reqs:
        if _n_times(target) >= 24 * len(req["month"]):
            print(f"skip {target.name} (complete)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".nc.part")
        print(f"requesting {target.name} ({len(req['month'])} months, ~{34 * len(req['month'])} MB) ...")
        client.retrieve(DATASET, req, str(part))
        part.rename(target)
        print(f"wrote {target}")
    return reqs


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def files(era5_dir):
    found = sorted(Path(era5_dir).glob("*.nc"))
    if not found:
        raise FileNotFoundError(f"no ERA5 NetCDF files in {era5_dir}; run clara-gapfill download-era5 "
                                "(or point paths.era5_dir at your monthly-by-hour ttr files)")
    return found


def month_blocks(path, months=None):
    """Yield (month Period, hourly OLR Dataset with 24 times, W m-2 positive upward) from one file; only the
    months in `months` (a set of Periods) when given."""
    with xr.open_dataset(path) as ds:
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        if "ttr" not in ds:
            raise ValueError(f"{path.name}: no 'ttr' variable; expected ERA5 top net thermal radiation")
        stamps = pd.to_datetime(ds[tname].values)
        stamp_months = (stamps - pd.Timedelta("1h")).to_period("M")  # the 24:00 stamp belongs to its month
        for month in stamp_months.unique():
            if months is not None and month not in months:
                continue
            idx = np.flatnonzero(stamp_months == month)
            if len(idx) != 24:
                raise ValueError(f"{path.name} {month}: {len(idx)} hourly fields, expected 24 (is this the "
                                 "'monthly averaged reanalysis by hour of day' product?)")
            sub = ds.isel({tname: idx}).drop_vars(["number", "expver"], errors="ignore").load()
            sub = grid.standardise_grid(sub)
            yield month, xr.Dataset({"olr": -sub["ttr"].astype(np.float64) / 3600.0})
