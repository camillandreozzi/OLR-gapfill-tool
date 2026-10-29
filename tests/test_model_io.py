"""Model files round-trip exactly, contain no pickle, and refuse grids they were not fitted for."""

import json
import zipfile

import numpy as np
import pytest

from clara_gapfill import config, demo, fit, model_io, predict


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    out = tmp_path_factory.mktemp("m")
    cfg = config.load(None, ["grid.res_deg=30", "time.first_month='2020-01'", "time.last_month='2022-12'",
                             "model.n_inducing=30", "model.restarts=1", "model.max_iter=15",
                             "era5.smoothing_radii_km=[0.0, 600.0]"])
    obs, grid_table, _ = demo.synthetic_tables(cfg, observed_fraction=0.5)
    model = fit.run(cfg, obs, out / "model", verbose=False)
    return cfg, obs, grid_table, model, out / "model"


def test_round_trip_predictions_are_identical(fitted):
    cfg, obs, g, model, path = fitted
    back, info = model_io.load(path)
    x = fit.inputs(g, info["origin"])
    e, s = g[info["era5_column"]].to_numpy(), g.post_gap.to_numpy()
    a, b = model.predict(x, e, np.ones(len(g)), s=s), back.predict(x, e, np.ones(len(g)), s=s)
    for k in ("mean", "latent_var", "obs_var", "mean_fn"):
        assert np.array_equal(a[k], b[k])
    assert back.coefficients() == model.coefficients()


def test_model_files_hold_no_pickle_and_record_the_grid(fitted):
    cfg, *_, path = fitted
    with zipfile.ZipFile(path / "model.npz") as z:
        for name in z.namelist():
            assert np.load(z.open(name), allow_pickle=False) is not None
    info = json.loads((path / "model.json").read_text())
    assert info["res_deg"] == 30 and info["kernel"] == "exponential" and info["gap_start"] == "2021-09"
    assert info["era5_column"] in ("era5_night_r0", "era5_night_r600")
    assert (path / "hyperparameters.csv").exists() and (path / "insample_fitted.csv").exists()


def test_predict_refuses_another_resolution_or_gap(fitted):
    cfg, *_, path = fitted
    _, info = model_io.load(path)
    with pytest.raises(ValueError, match="30 deg"):
        predict.check_compatible(info, config.load(None, ["grid.res_deg=5"]))
    with pytest.raises(ValueError, match="gap"):
        predict.check_compatible(info, config.load(None, ["grid.res_deg=30", "time.gap_start='2022-01'"]))
    predict.check_compatible(info, cfg)
