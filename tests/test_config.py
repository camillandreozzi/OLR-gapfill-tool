"""Defaults, user files, --set overrides and validation."""

import pytest

from clara_gapfill import config


def test_defaults_are_the_winning_model():
    cfg = config.load()
    m = cfg["model"]
    assert (m["kernel"], m["n_inducing"], m["restarts"], m["gap_step"]) == ("exponential", 1000, 3, True)
    assert cfg["grid"]["res_deg"] == 5 and cfg["era5"]["radius_km"] == "auto"
    assert cfg["qc"]["olr_min_from"] == {2024: 75.0, 2025: 90.0}


def test_user_file_overrides_and_relative_paths(tmp_path):
    (tmp_path / "c.toml").write_text('[grid]\nres_deg = 1\n[paths]\nwork_dir = "w"\n[model]\nrestarts = 1\n')
    cfg = config.load(tmp_path / "c.toml", ["time.last_month=2025-12", "model.kernel=matern32"])
    assert cfg["grid"]["res_deg"] == 1 and cfg["model"]["restarts"] == 1
    assert cfg["time"]["last_month"] == "2025-12" and cfg["model"]["kernel"] == "matern32"
    assert cfg["paths"]["work_dir"] == str((tmp_path / "w").resolve())


@pytest.mark.parametrize("item", ["grid.res_deg=7", "grid.res_deg=0", "grid.res_deg=2.5", "model.kernel=rbf",
                                  "time.first_month=2027-01", "era5.radius_km=700", "model.restarts=4",
                                  "predict.table_format=xlsx", "validate.schemes=['spatial','loo']"])
def test_invalid_settings_are_refused(item):
    with pytest.raises(config.ConfigError):
        config.load(None, [item])


def test_unknown_settings_are_refused(tmp_path):
    (tmp_path / "c.toml").write_text("[grid]\nresolution = 1\n")
    with pytest.raises(config.ConfigError, match="grid.resolution"):
        config.load(tmp_path / "c.toml")
    with pytest.raises(config.ConfigError):
        config.load(None, ["model.n_inducing"])


@pytest.mark.parametrize("res", [1, 2, 3, 5, 10, 30])
def test_allowed_resolutions(res):
    assert config.load(None, [f"grid.res_deg={res}"])["grid"]["res_deg"] == res


def test_override_values_parse_as_toml():
    assert config.parse_override("era5.smoothing_radii_km=[0.0, 600.0]") == {"era5": {"smoothing_radii_km": [0.0, 600.0]}}
    assert config.parse_override("maps.months=observed") == {"maps": {"months": "observed"}}
    assert config.parse_override("model.gap_step=false") == {"model": {"gap_step": False}}


def test_section_hash_tracks_only_the_given_settings():
    a, b = config.load(), config.load(None, ["maps.dpi=200"])
    assert config.section_hash(a, ["model", "grid"]) == config.section_hash(b, ["model", "grid"])
    assert config.section_hash(a, ["maps"]) != config.section_hash(b, ["maps"])
