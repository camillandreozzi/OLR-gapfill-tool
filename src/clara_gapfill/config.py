"""Configuration: the packaged defaults (default_config.toml), a user TOML file on top, and --set overrides.

    cfg = config.load("my_config.toml", ["grid.res_deg=1", "model.restarts=1"])
    cfg["grid"]["res_deg"]

A user file may only contain settings that exist in the defaults (typos are errors, not silently ignored).
Relative paths in [paths] are resolved against the folder of the user file (the current folder without one).
"""

import hashlib
import json
import os
import tomllib
from importlib import resources
from pathlib import Path

import pandas as pd

KERNELS = ("exponential", "gaussian", "matern32")
SCHEMES = ("spatial", "temporal", "joint")
FREE_FORM = {("qc", "olr_min_from")}  # tables whose keys are data, not setting names


class ConfigError(ValueError):
    pass


def default_text():
    return resources.files("clara_gapfill").joinpath("default_config.toml").read_text()


def defaults():
    return tomllib.loads(default_text())


def _merge(base, over, prefix=()):
    for key, value in over.items():
        where = ".".join(prefix + (key,))
        if key not in base:
            raise ConfigError(f"unknown setting {where!r} (see clara-gapfill init-config for the full list)")
        if isinstance(base[key], dict) and prefix + (key,) not in FREE_FORM:
            if not isinstance(value, dict):
                raise ConfigError(f"{where!r} is a section, not a value")
            _merge(base[key], value, prefix + (key,))
        else:
            base[key] = value


def parse_override(item):
    """'section.key=value' -> ({section: {key: value}}); value is parsed as TOML, else kept as a string."""
    if "=" not in item:
        raise ConfigError(f"--set expects section.key=value, got {item!r}")
    key, text = (s.strip() for s in item.split("=", 1))
    try:
        value = tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        value = text
    out = node = {}
    parts = key.split(".")
    for p in parts[:-1]:
        node[p] = {}
        node = node[p]
    node[parts[-1]] = value
    return out


def load(path=None, overrides=()):
    """Defaults <- user file <- overrides, validated, with absolute paths."""
    cfg = defaults()
    base = Path.cwd()
    if path is not None:
        path = Path(path)
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
        base = path.resolve().parent
    for item in overrides:
        _merge(cfg, parse_override(item))
    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        cfg["paths"][key] = str(p if p.is_absolute() else (base / p).resolve())
    validate(cfg)
    cfg["_source"] = str(path) if path is not None else "defaults"
    return cfg


def _month(cfg, key):
    try:
        return pd.Period(cfg["time"][key], "M")
    except (ValueError, TypeError) as err:
        raise ConfigError(f"time.{key} = {cfg['time'][key]!r} is not a month (YYYY-MM)") from err


def validate(cfg):
    res = cfg["grid"]["res_deg"]
    if isinstance(res, float) and res.is_integer():
        res = cfg["grid"]["res_deg"] = int(res)
    if not isinstance(res, int) or isinstance(res, bool) or res < 1 or 180 % res:
        allowed = ", ".join(str(r) for r in range(1, 91) if 180 % r == 0)
        raise ConfigError(f"grid.res_deg = {res!r}: must be an integer that divides 180 ({allowed})")
    first, last = _month(cfg, "first_month"), _month(cfg, "last_month")
    gap_start, gap_end = _month(cfg, "gap_start"), _month(cfg, "gap_end")
    if first > last:
        raise ConfigError("time.first_month is after time.last_month")
    if gap_start > gap_end:
        raise ConfigError("time.gap_start is after time.gap_end")
    pd.Timestamp(cfg["time"]["origin"])
    cfg["qc"]["olr_min_from"] = {int(k): float(v) for k, v in cfg["qc"]["olr_min_from"].items()}
    radii = cfg["era5"]["smoothing_radii_km"]
    if not radii or any(r < 0 for r in radii):
        raise ConfigError("era5.smoothing_radii_km must be a non-empty list of radii >= 0")
    radius = cfg["era5"]["radius_km"]
    if radius != "auto" and (isinstance(radius, str) or float(radius) not in [float(r) for r in radii]):
        raise ConfigError(f"era5.radius_km = {radius!r}: use 'auto' or one of era5.smoothing_radii_km {radii}")
    m = cfg["model"]
    if m["kernel"] not in KERNELS:
        raise ConfigError(f"model.kernel = {m['kernel']!r}: expected one of {KERNELS}")
    if not 1 <= m["restarts"] <= 3:
        raise ConfigError("model.restarts must be 1, 2 or 3")
    if m["n_inducing"] < 1:
        raise ConfigError("model.n_inducing must be >= 1")
    if cfg["predict"]["table_format"] not in ("csv", "parquet"):
        raise ConfigError("predict.table_format must be 'csv' or 'parquet'")
    if cfg["predict"]["batch_rows"] < 1:
        raise ConfigError("predict.batch_rows must be >= 1")
    if cfg["predict"]["n_overpasses_for_sd"] <= 0:
        raise ConfigError("predict.n_overpasses_for_sd must be > 0")
    months = cfg["maps"]["months"]
    if not (months in ("all", "observed") or isinstance(months, list)):
        raise ConfigError("maps.months must be 'all', 'observed' or a list of 'YYYY-MM'")
    bad = [s for s in cfg["validate"]["schemes"] if s not in SCHEMES]
    if bad:
        raise ConfigError(f"validate.schemes: unknown {bad}; expected a subset of {SCHEMES}")
    return cfg


def months(cfg):
    """Every month of [time.first_month, time.last_month]."""
    return pd.period_range(cfg["time"]["first_month"], cfg["time"]["last_month"], freq="M")


def gp_settings(cfg):
    """The [model] section in the flat form gp.GP expects."""
    return {k: v for k, v in cfg["model"].items() if k not in ("kernel", "threads")}


def section_hash(cfg, keys):
    """Short SHA-256 of the given settings ('section' or 'section.key'), for the stage manifests."""
    picked = {}
    for k in keys:
        sec, _, name = k.partition(".")
        picked[k] = cfg[sec][name] if name else cfg[sec]
    text = json.dumps(picked, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save(cfg, path):
    """Write the settings that produced an output next to it, as JSON (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cfg, indent=2, default=str))
    os.replace(tmp, path)
