"""The pipeline stages and their bookkeeping.

    clara    raw CLARA files            -> work/clara_geolocated.parquet
    era5     ERA5 by-hour files         -> work/era5_night_1deg.nc
    tables   both of the above          -> work/res<N>/training_table.parquet, prediction_grid.parquet
    fit      training table             -> output/res<N>/<kernel>/model/
    predict  model + prediction grid    -> output/res<N>/<kernel>/predictions/
    maps     predictions                -> output/res<N>/<kernel>/maps/
    validate training table             -> output/res<N>/<kernel>/validation/  (optional, slow)

Each stage writes a manifest (a fingerprint of its input files, of the upstream stages and of the settings it
depends on) and is skipped when the fingerprint is unchanged, so `clara-gapfill run` only redoes what changed.
--force re-runs a stage regardless.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import xarray as xr

from . import __version__, clara, era5, era5_night, fit, maps, model_io, predict, samples, tables, validate
from .config import months as config_months, save as save_config, section_hash
from .layout import Layout

STAGE_SETTINGS = {
    "clara": ["qc.geo_max_gap_s", "time.first_month", "time.last_month"],
    "era5": ["era5.night_sza_min", "era5.fallback_hours", "time.first_month", "time.last_month"],
    "tables": ["grid", "qc", "era5.smoothing_radii_km", "time"],
    "fit": ["model", "era5.radius_km", "time.origin", "time.gap_start"],
    "predict": ["predict", "time.gap_start"],
    "maps": ["maps"],
}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def _signature(paths):
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in map(Path, paths)]


def fingerprint(cfg, stage, upstream=(), files=()):
    text = json.dumps({"version": __version__, "settings": section_hash(cfg, STAGE_SETTINGS[stage]),
                       "upstream": list(upstream), "files": _signature(files)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def manifest_path(output):
    output = Path(output)
    return output / "manifest.json" if output.suffix == "" else output.with_name(output.name + ".manifest.json")


def read_fingerprint(output):
    p = manifest_path(output)
    return json.loads(p.read_text())["fingerprint"] if p.exists() else None


def current(output, fp, force):
    if force or read_fingerprint(output) != fp:
        return False
    return Path(output).exists()


def write_manifest(output, fp, **info):
    p = manifest_path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": fp, "tool_version": __version__} | info, indent=2, default=str))


def _need(output, what):
    if not Path(output).exists():
        raise FileNotFoundError(f"{output} is missing; run clara-gapfill {what} first")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_clara(cfg, force=False):
    lay = Layout(cfg)
    src = Path(cfg["paths"]["clara_dir"])
    files = clara.source_files(src) if src.exists() else []
    if not files:
        raise FileNotFoundError(f"no CLARA files in {src} (expected CLARA_OLR*.tar.gz and CLARA_lat_lon_*.tar.gz "
                                "or the extracted .save / .csv files); set paths.clara_dir")
    fp = fingerprint(cfg, "clara", files=files)
    if current(lay.geolocated, fp, force):
        print(f"clara: up to date ({lay.geolocated})")
        return
    first, last = config_months(cfg)[[0, -1]]
    print(f"clara: reading {len(files)} CLARA files from {src} ...", flush=True)
    df = clara.read_clara(src, lay.extracted, cfg["qc"]["geo_max_gap_s"], set(range(first.year, last.year + 1)))
    lay.work.mkdir(parents=True, exist_ok=True)
    df.to_parquet(lay.geolocated, index=False)
    write_manifest(lay.geolocated, fp, samples=len(df), first=df.time.min(), last=df.time.max())
    print(f"clara: {len(df):,} samples {df.time.min():%Y-%m-%d} .. {df.time.max():%Y-%m-%d}, "
          f"{int(df.earth_view.sum()):,} Earth viewing -> {lay.geolocated}")


def stage_era5(cfg, force=False):
    lay = Layout(cfg)
    files = era5.files(cfg["paths"]["era5_dir"])
    fp = fingerprint(cfg, "era5", files=files)
    if current(lay.era5_night, fp, force):
        print(f"era5: up to date ({lay.era5_night})")
        return
    months = list(config_months(cfg))
    print(f"era5: building the night-time ERA5 product from {len(files)} file(s) ...", flush=True)
    ds, checks = era5_night.build(cfg["paths"]["era5_dir"], months, cfg["era5"]["night_sza_min"],
                                  int(cfg["era5"]["fallback_hours"]))
    lay.work.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(lay.era5_night)
    checks.to_csv(lay.work / "era5_night_checks.csv", index=False, float_format="%.4f")
    write_manifest(lay.era5_night, fp, months=ds.sizes["time"])
    print(f"era5: {ds.sizes['time']} months -> {lay.era5_night}")


def stage_tables(cfg, force=False):
    lay = Layout(cfg)
    _need(lay.geolocated, "prepare --only clara")
    _need(lay.era5_night, "prepare --only era5")
    fp = fingerprint(cfg, "tables", upstream=[read_fingerprint(lay.geolocated), read_fingerprint(lay.era5_night)])
    if current(lay.training_table, fp, force) and lay.prediction_grid.exists():
        print(f"tables: up to date ({lay.work_res})")
        return
    print(f"tables: CLARA cell-months and the prediction grid at {cfg['grid']['res_deg']} deg ...", flush=True)
    geo = pd.read_parquet(lay.geolocated)
    with xr.open_dataset(lay.era5_night) as night:
        s, obs, gridt = tables.build(cfg, geo, night.load())
    lay.work_res.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(lay.training_table, index=False)
    gridt.to_parquet(lay.prediction_grid, index=False)
    lay.preprocessing.mkdir(parents=True, exist_ok=True)
    qc = samples.qc_counts(s, cfg["qc"])
    val = tables.validation_checks(obs, gridt, cfg)
    qc.to_csv(lay.preprocessing / "qc_counts.csv", index=False)
    val.to_csv(lay.preprocessing / "validation_checks.csv", index=False)
    tables.observations_summary(obs).to_csv(lay.preprocessing / "observations_summary.csv")
    save_config(cfg, lay.preprocessing / "config_used.json")
    write_manifest(lay.training_table, fp, observations=len(obs), grid_rows=len(gridt))
    print(qc.to_string(index=False))
    print(val.to_string(index=False))
    print(f"tables: {len(obs):,} observed cell-months, {len(gridt):,} prediction-grid cell-months -> {lay.work_res}")


def load_training(cfg, table=None):
    """(training table, fingerprint of its source) from the pipeline or from a user table."""
    if table is not None:
        return tables.normalise(tables.read_table(table), cfg, "training"), _signature([table])
    lay = Layout(cfg)
    _need(lay.training_table, "prepare")
    return pd.read_parquet(lay.training_table), [read_fingerprint(lay.training_table)]


def stage_fit(cfg, force=False, table=None, verbose=True):
    lay = Layout(cfg)
    obs, up = load_training(cfg, table)
    fp = fingerprint(cfg, "fit", upstream=up)
    if current(lay.model, fp, force) and (lay.model / "model.json").exists():
        print(f"fit: up to date ({lay.model})")
        return
    fit.run(cfg, obs, lay.model, verbose=verbose)
    save_config(cfg, lay.model / "config_used.json")
    write_manifest(lay.model, fp, table=str(table or lay.training_table))


def stage_predict(cfg, force=False, model_dir=None, grid_path=None, out_dir=None):
    lay = Layout(cfg)
    model_dir = Path(model_dir) if model_dir else lay.model
    model, info = model_io.load(model_dir)
    predict.check_compatible(info, cfg, user_grid=grid_path is not None)
    out_dir = Path(out_dir) if out_dir else lay.out_res / info["kernel"] / "predictions"
    oos_path = model_dir.parent / "validation" / "oos_predictions_spatial.csv"
    if grid_path is not None:
        gridt, up = tables.normalise(tables.read_table(grid_path), cfg, "grid"), _signature([grid_path])
    else:
        _need(lay.prediction_grid, "prepare")
        gridt, up = pd.read_parquet(lay.prediction_grid), [read_fingerprint(lay.training_table)]
    files = [model_dir / "model.npz", model_dir / "model.json"] + ([oos_path] if oos_path.exists() else [])
    fp = fingerprint(cfg, "predict", upstream=up, files=files)
    if current(out_dir, fp, force):
        print(f"predict: up to date ({out_dir})")
        return out_dir
    oos = None
    if oos_path.exists():
        oos = pd.read_csv(oos_path, usecols=["time", "lat", "lon", "residual"], parse_dates=["time"])
        oos["time"] = oos.time.astype("datetime64[ns]")
    print(f"predict: {len(gridt):,} cell-months with the {info['kernel']} model from {model_dir} ...", flush=True)
    g, x = predict.predict_grid(model, info, gridt, cfg["predict"]["n_overpasses_for_sd"], oos,
                                cfg["predict"]["batch_rows"])
    predict.write(g, x, model, info, cfg, out_dir, complete=grid_path is None)
    save_config(cfg, out_dir / "config_used.json")
    write_manifest(out_dir, fp, kernel=info["kernel"], model=str(model_dir),
                   grid=str(grid_path or lay.prediction_grid))
    return out_dir


def read_predictions(pred_dir):
    found = sorted(Path(pred_dir).glob("grid_predictions.*"))
    if not found:
        raise FileNotFoundError(f"no grid_predictions table in {pred_dir}; run clara-gapfill predict first")
    g = tables.read_table(found[0])
    g["time"] = pd.to_datetime(g.time).astype("datetime64[ns]")
    for col in ("clara_observed", "night_fallback"):
        g[col] = g[col].astype(bool)
    return g


def stage_maps(cfg, force=False, pred_dir=None):
    lay = Layout(cfg)
    pred_dir = Path(pred_dir) if pred_dir else lay.predictions
    if not cfg["maps"]["enabled"]:
        print("maps: disabled (maps.enabled = false)")
        return
    out_dir = pred_dir.parent / "maps"
    fp = fingerprint(cfg, "maps", upstream=[read_fingerprint(pred_dir)])
    if current(out_dir, fp, force):
        print(f"maps: up to date ({out_dir})")
        return
    g = read_predictions(pred_dir)
    mf = manifest_path(pred_dir)
    kernel = json.loads(mf.read_text()).get("kernel", cfg["model"]["kernel"]) if mf.exists() else cfg["model"]["kernel"]
    print(f"maps: {g.time.nunique()} months, maps.months = {cfg['maps']['months']!r} ...", flush=True)
    years = ((year, predict.to_dataset(gy, cfg["grid"]["res_deg"])) for year, gy in g.groupby("year"))
    n = maps.monthly_maps(g, years, cfg, kernel, out_dir)
    gm_path = pred_dir / "global_mean_monthly.csv"
    if gm_path.exists():
        maps.global_mean_figure(pd.read_csv(gm_path), kernel, out_dir / "global_mean_monthly.png", cfg["maps"]["dpi"])
        n += 1
    write_manifest(out_dir, fp, figures=n)
    print(f"maps: {n} figures -> {out_dir}")


def stage_validate(cfg, table=None, schemes=None, folds=None, verbose=False):
    lay = Layout(cfg)
    obs, _ = load_training(cfg, table)
    print(f"validate: blocked CV of the {cfg['model']['kernel']} GP on {len(obs):,} cell-months "
          f"(M = {cfg['model']['n_inducing']}, {cfg['model']['restarts']} start(s) per fold) ...", flush=True)
    summary, _ = validate.run(cfg, obs, lay.validation, schemes, folds, verbose)
    save_config(cfg, lay.validation / "config_used.json")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary[["cv_type", "folds", "mean_fold_rmse"] + [c for c in summary if c in
                      ("pooled_rmse", "pooled_cover95", "pooled_crps")]].to_string(index=False))
    print(f"validate: results in {lay.validation} (predict now adds out-of-sample residual maps)")


def run_all(cfg, force=False, verbose=True):
    stage_clara(cfg, force)
    stage_era5(cfg, force)
    stage_tables(cfg, force)
    stage_fit(cfg, force, verbose=verbose)
    stage_predict(cfg, force)
    stage_maps(cfg, force)

