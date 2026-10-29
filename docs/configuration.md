# Configuration

Every setting has a default in [`src/clara_gapfill/default_config.toml`](../src/clara_gapfill/default_config.toml),
which also explains it in a comment. `clara-gapfill init-config my_run.toml` copies that file for you to edit.
Settings are resolved in this order, later ones winning:

1. the packaged defaults;
2. your file (`-c my_run.toml`), which may contain only the settings you change;
3. `--set section.key=value` on the command line (repeatable). Values are parsed as TOML: `--set
   model.gap_step=false`, `--set "maps.months=['2021-01']"`.

Unknown or misspelt settings are errors. Relative paths in `[paths]` are resolved against the folder of your
config file. `clara-gapfill show-config -c my_run.toml` prints the resolved settings. Every output folder
gets a `config_used.json`.

## Sections

**`[paths]`** sets where the inputs are and where intermediate files and outputs go. The work and output
folders are organised by resolution (`res<N>/`) and kernel, so several runs can share them.

**`[grid]`** has one setting, `res_deg`, the cell size of both the training cell-months and the
predictions: 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, … (any integer that divides 180). ERA5 is always
processed at 1° and then averaged onto these cells. What to expect:

| Resolution | Observed cell-months (2020-01 to 2026-09) | Grid rows | Notes |
|---|---|---|---|
| 10° | 14.1 k | 52.5 k | quick to fit; coarse maps |
| 5° (reference) | 37.9 k | 210 k | the resolution the model was selected at |
| 1° | 188.5 k | 5.25 M | fit time grows linearly with rows; 82 % of cell-months hold a single 30 s sample (22 % at 5°), so the 1° detail comes from ERA5 through β₁ |

The model's length scales are in chords, which do not depend on the grid. The CV block sizes and buffers are
in degrees and km, so they apply unchanged at any resolution.

**`[time]`** sets the period (`first_month`, `last_month`), the origin of the long-term time coordinate, and
the CLARA gap (`gap_start`, `gap_end`). For a period that does not straddle the gap, set
`model.gap_step = false`, because a step needs data on both sides. Months ERA5 has not yet published are left
out of the grid with a warning.

**`[qc]`** holds the CLARA instrument constants and the QC thresholds (see [data.md](data.md)).

**`[era5]`** holds the night definition, the polar-day fallback, the cap-smoothing radii and the choice of
radius (`"auto"` or one of the radii).

**`[model]`** has the kernel, the gap step, the sparse approximation (`n_inducing`), the optimiser
(`restarts`, `max_iter`), the CPU `threads`, `chunk_size` and the initial values. The defaults reproduce the
reference fit. For a quicker fit use `n_inducing = 500` and `restarts = 1`: in the reference benchmark,
predictions changed by less than 0.2 W m⁻² sr⁻¹ RMSE between *M* = 250 and 2000.

**`[predict]`** sets:
- the number of overpasses for `pred_sd_obs`;
- the format of the grid table (`csv` or `parquet`);
- whether to write yearly NetCDF files, per-month CSV files and the global mean;
- `batch_rows`, which caps memory.

**`[maps]`** sets which months to draw (`"all"`, `"observed"` or a list), the projection and the dpi.

**`[validate]`** sets the blocked cross-validation: block sizes, windows, buffers and the number of joint
folds.

## Staleness

Each stage records a fingerprint of its inputs and of the settings it depends on. `run` skips stages whose
fingerprint is unchanged:

| Stage | Depends on |
|---|---|
| clara | the CLARA files, `qc.geo_max_gap_s`, the period |
| era5 | the ERA5 files, `era5.night_sza_min`, `era5.fallback_hours`, the period |
| tables | the two stages above, `[grid]`, `[qc]`, `era5.smoothing_radii_km`, `[time]` |
| fit | the training table, `[model]`, `era5.radius_km`, `time.origin`, `time.gap_start` |
| predict | the model, the prediction grid, `[predict]`, `time.gap_start` |
| maps | the predictions, `[maps]` |

`--force` re-runs a stage regardless.
