# Outputs

Everything goes under `paths.work_dir` (intermediate files) and `paths.output_dir` (results), organised by
grid resolution and kernel:

```
work/
  extracted/                          CLARA archives, extracted
  clara_geolocated.parquet            every CLARA sample with its footprint geometry
  era5_night_1deg.nc                  ERA5 night-time OLR (and 24 h mean, night fraction, fallback flag), 1 deg
  era5_night_checks.csv               global means and polar-day cell counts per month
  res5/training_table.parquet         one row per observed cell-month
  res5/prediction_grid.parquet        every cell of every month with the ERA5 predictors
output/res5/
  preprocessing/                      qc_counts.csv, validation_checks.csv, observations_summary.csv
  exponential/
    model/                            model.npz, model.json, hyperparameters.csv, insample_fitted.csv
    predictions/                      grid_predictions.csv (or .parquet), netcdf/<yyyy>_predictions.nc,
                                      global_mean_monthly.csv, [monthly_csv/]
    maps/<yyyy>/                      <yyyy>_<mm>_predicted_olr.png, _uncertainty.png,
                                      _observed_olr.png, _residuals.png (months with CLARA data)
    maps/global_mean_monthly.png
    validation/                       folds.csv, metrics_by_fold.csv, metrics_summary.csv,
                                      oos_predictions_<scheme>.csv, folds/ (resume state)
```

Every folder also has `config_used.json` (the resolved settings) and `manifest.json` (the stage fingerprint).

## `grid_predictions` (one row per cell-month, sorted by time, lat, lon)

Values are in W m⁻² sr⁻¹ (CLARA radiance) unless stated otherwise.

| Column | Meaning |
|---|---|
| `year`, `month`, `time` | the month (`time` = first day) |
| `lat`, `lon` | cell centre (deg) |
| `pred_mean` | **gap-filled monthly-mean CLARA night-side OLR**: the GP posterior mean |
| `latent_sd` | posterior SD of `pred_mean` (the monthly mean itself) |
| `pred_sd_obs` | SD of a new CLARA cell-month mean from `predict.n_overpasses_for_sd` overpasses (adds the sampling noise) |
| `mean_fn` | β₀ + β₁ ERA5 + β₂ post_gap alone (what the GP adds is `pred_mean − mean_fn`) |
| `era5_night` | the ERA5 predictor used (W m⁻², smoothed at the model's radius) |
| `post_gap` | 1 from `time.gap_start` on |
| `clara_observed` | 1 where CLARA observed the cell-month |
| `clara`, `n_overpasses` | the observation, where there is one |
| `night_fallback` | 1 in polar-day cells: the predictor is the darkest-hours mean, so the value is extrapolation |
| `residual_insample` | `clara − pred_mean`; the observation was part of the fit |
| `residual_oos_spatial` | the same from spatial block CV, after `clara-gapfill validate` (else empty) |
| `cell_area_km2` | cell area |

The yearly NetCDF files hold the same variables on (time, lat, lon) with units and long names.

## `global_mean_monthly.csv`

| Column | Meaning |
|---|---|
| `global_mean` | area-weighted mean of `pred_mean` over all cells |
| `sd_full_covariance` | its SD from the full posterior covariance (use this one) |
| `sd_if_independent`, `sd_mean_of_variances` | naive alternatives, for comparison only |
| `polar_day_area_fraction` | area share of fallback (extrapolated) cells |

## `model/`

- `model.json` holds the kernel, the settings, the fitted hyperparameters in physical units and the bound. It
  also records what a prediction grid must match: `res_deg`, `era5_column`, `origin` and `gap_start`.
- `model.npz` holds the arrays.
- `hyperparameters.csv` holds the same parameters as one row, plus the correlation of each ERA5 radius with
  CLARA.
- `insample_fitted.csv` holds the fitted values at every training row.

## `validation/`

- `metrics_by_fold.csv` has the RMSE, area-weighted RMSE, MAE, bias, NLPD, CRPS, 95 % interval width, R²
  and the 50/80/95 % coverage, for every fold.
- `metrics_summary.csv` has the mean per-fold RMSE per scheme, the pooled metrics and the primary score (the
  mean over the schemes).
