"""End to end on synthetic data through the CLI: fit, predict, maps, the up-to-date check and a validation fold."""

import json

import numpy as np
import pandas as pd

from clara_gapfill import cli, config, demo, pipeline
from clara_gapfill.layout import Layout


def test_demo_end_to_end(tmp_path, capsys):
    info, rmse = demo.run(tmp_path, res=30, quick=True)
    out = tmp_path / "output" / "res30" / "exponential"
    table = pd.read_csv(out / "predictions" / "grid_predictions.csv")
    assert len(table) == 48 * 6 * 12 and table.pred_mean.notna().all() and (table.latent_sd > 0).all()
    assert (out / "predictions" / "netcdf" / "2021_predictions.nc").exists()
    assert (out / "maps" / "2020" / "2020_07_predicted_olr.png").exists()
    assert (out / "maps" / "global_mean_monthly.png").exists()
    assert rmse < 15 and 0.3 < info["hyperparameters"]["beta1"] < 0.8

    # a second predict with the same inputs and settings is skipped; a changed setting re-runs it
    cfg = demo.demo_config(tmp_path.resolve(), 30, True)
    capsys.readouterr()
    pipeline.stage_predict(cfg)
    assert "up to date" in capsys.readouterr().out
    cfg2 = demo.demo_config(tmp_path.resolve(), 30, True, ["predict.table_format=parquet"])
    pipeline.stage_predict(cfg2)
    assert (out / "predictions" / "grid_predictions.parquet").exists()
    assert not (out / "predictions" / "grid_predictions.csv").exists()


def test_cli_fit_and_predict_user_tables(tmp_path):
    cfg = config.load(None, ["grid.res_deg=30", "time.first_month='2020-01'", "time.last_month='2022-12'",
                             "era5.smoothing_radii_km=[0.0, 600.0]"])
    obs, g, _ = demo.synthetic_tables(cfg)
    obs[["lat", "lon", "year", "month", "clara", "n_overpasses", "era5_night_r600"]].to_csv(tmp_path / "t.csv", index=False)
    g[["lat", "lon", "year", "month", "era5_night_r600", "clara"]].to_csv(tmp_path / "g.csv", index=False)
    common = ["--set", "grid.res_deg=30", "--set", "model.n_inducing=30", "--set", "model.restarts=1",
              "--set", "model.max_iter=10", "--set", f"paths.output_dir='{tmp_path / 'out'}'",
              "--set", f"paths.work_dir='{tmp_path / 'work'}'", "--set", "era5.smoothing_radii_km=[0.0, 600.0]",
              "-q"]
    cli.main(["fit", "--table", str(tmp_path / "t.csv")] + common)
    model_dir = tmp_path / "out" / "res30" / "exponential" / "model"
    assert json.loads((model_dir / "model.json").read_text())["era5_column"] == "era5_night_r600"
    cli.main(["predict", "--grid", str(tmp_path / "g.csv"), "--model", str(model_dir),
              "--out", str(tmp_path / "pred")] + common)
    p = pd.read_csv(tmp_path / "pred" / "grid_predictions.csv")
    assert len(p) == len(g) and p.pred_mean.notna().all()
    assert np.isfinite(p.residual_insample[p.clara_observed == 1]).all()


def test_validate_one_fold(tmp_path):
    cfg = config.load(None, ["grid.res_deg=30", "time.first_month='2020-01'", "time.last_month='2022-12'",
                             "model.n_inducing=30", "model.restarts=1", "model.max_iter=10",
                             "era5.smoothing_radii_km=[0.0, 600.0]", f"paths.output_dir='{tmp_path}'",
                             f"paths.work_dir='{tmp_path / 'work'}'"])
    obs, _, _ = demo.synthetic_tables(cfg, observed_fraction=0.6)
    lay = Layout(cfg)
    lay.work_res.mkdir(parents=True)
    obs.to_parquet(lay.training_table)
    pipeline.stage_validate(cfg, schemes=["spatial"], folds=[0])
    m = pd.read_csv(lay.validation / "metrics_by_fold.csv")
    assert list(m.cv_type) == ["spatial"] and m.rmse.iloc[0] > 0
    assert (lay.validation / "folds" / "spatial_00.parquet").exists()
