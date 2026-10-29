# Input data

The tool reads the CLARA files in the format the instrument team distributes them, and the ERA5 product it
downloads itself. None of these data are part of the repository.

## CLARA (`paths.clara_dir`)

CLARA is a 30 s along-track radiometer series, not a gridded product. It comes as two kinds of files. Both can
be given as `.tar.gz` archives, as extracted files anywhere under `clara_dir`, or both.

### OLR: `CLARA_OLR*.tar.gz` → `CLARA_OLR_<yyyy>_<m>.save`

One IDL save file per month, holding a structure `OLR_ALL` with:

| Field | Content |
|---|---|
| `JDAY` | Julian date (UTC) of each sample |
| `OLR` | OLR radiance in W m⁻² sr⁻¹, as measured (not angle-corrected) |

The channel also sees the Sun: values of about 10⁴–9·10⁴ are sun-viewing periods, which the plausible-range
QC removes. A month found in several places is taken from the last one read:
- a later archive in name order, so `CLARA_OLR_2025.tar.gz` overrides `CLARA_OLR.tar.gz`;
- a loose `.save` file overrides any archive.

### Geolocation: `CLARA_lat_lon_<yyyy>.tar.gz` → `CLARA_lon_lat_alt-<yyyy>-<m>-<d>_V50_1_4[_test].csv`

One CSV per day at 1 min cadence, with a header line such as
`# datetime,z_vector_satellite_position_angles,satellite_lat_long,clara_lat_long`:

| Column | Content |
|---|---|
| `datetime` | time of day (UTC) |
| `z_vector_satellite_position_angles` | angle of the line of sight from nadir (deg) |
| `satellite_lat_long` | `"(lat, lon, alt_m)"` of the satellite, WGS84 geodetic |
| `clara_lat_long` | `"(lat, lon, alt_m)"` of the footprint; `"(0, 0, 0)"` when looking away from the Earth, empty when the line of sight passes above the limb |

A final file is used rather than a `_test` file of the same day. The footprint is recomputed as the near
intersection of the line of sight with the 6371 km sphere; some older releases stored the far intersection.
The recomputation also gives the viewing zenith angle (VZA). Positions are interpolated onto the OLR sample
times on the unit sphere, and only between geolocation rows at most `qc.geo_max_gap_s` apart.

Archives are extracted into `<work_dir>/extracted/`; the tool never writes into `clara_dir`. Loose files
inside folders named `extracted` under `clara_dir` are ignored, so an old extraction there is not read
twice.

### QC (`[qc]`)

A sample enters the model when all of these hold:
- it is Earth viewing;
- its OLR is within [`olr_min`, `olr_max`], where `olr_min` is raised from the years in `olr_min_from` on;
- its footprint is on the night side (SZA > `night_sza_min`);
- VZA ≤ `vza_max`.

QC'd samples more than `overpass_gap_min` apart start a new overpass. `output/res<N>/preprocessing/qc_counts.csv`
gives the attrition at each stage.

## ERA5 (`paths.era5_dir`)

`clara-gapfill download-era5` downloads `top_net_thermal_radiation` (ttr) from
`reanalysis-era5-single-levels-monthly-means`, product type `monthly_averaged_reanalysis_by_hour_of_day`:

- native 0.25° grid;
- one NetCDF file per year, `era5_ttr_monthly_by_hour_<yyyy>.nc`, about 34 MB per month;
- for each month, 24 fields: the monthly mean of the ttr accumulation over the hour ending at each UTC hour.

Any NetCDF files of this product in `era5_dir` are read, so files you downloaded yourself work too. They
must be on the native grid, with 24 hourly fields per month.

The night-time product is built per month on 1° cells:
1. OLR = −ttr/3600.
2. Each hour is weighted by the fraction of the month's days on which that hour is night (solar zenith angle
   > `era5.night_sza_min` at the cell centre).
3. Polar-day cells, which have no night hour, use the mean of the `era5.fallback_hours` darkest hours and are
   flagged `night_fallback`.
4. The field is smoothed over spherical caps at each `era5.smoothing_radii_km`.
5. It is area-averaged onto the `grid.res_deg` cells.

## Your own tables

**For `fit --table`.** One row per observed cell-month; CSV or Parquet:

| Column | Required | Meaning |
|---|---|---|
| `lat`, `lon` | yes | cell centre (deg; lon in [−180, 180)) |
| `time`, or `year` + `month` | yes | the month (any date within it) |
| `clara` | yes | cell-month mean CLARA OLR (W m⁻² sr⁻¹) |
| `n_overpasses` | yes | number of distinct overpasses behind `clara` (≥ 1) |
| `era5_night` or `era5_night_r<R>` | yes | ERA5 night-time OLR (W m⁻²). Several `_r<R>` columns let `era5.radius_km = "auto"` choose. |
| `post_gap`, `model_time`, `cell_area_km2`, `period`, `night_fallback` | no | derived from the config when missing |

**For `predict --grid`.** The same without `clara` and `n_overpasses`. If `clara` is present, it is used for
the residual columns. The grid needs the ERA5 column the model was fitted with (recorded in `model.json`),
or a single `era5_night` column.
