"""Save and load fitted models without pickle: model.npz (arrays) and model.json (everything else).

model.npz   z (M, 6) inducing inputs; post_l, post_lb (M, M) Cholesky factors and post_c (M,) of the posterior;
            raw_<name> the fitted parameters (log of the positive ones, standardised units)
model.json  format_version, kernel, settings, standardisation statistics, the hyperparameters in physical
            units, and what the inputs must match: res_deg, era5_column, era5_radius_km, origin, gap_start

No training data are stored. import_pt() converts a full_model.pt of the original research pipeline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import __version__, gp

FORMAT_VERSION = 1


def save(model, out_dir, meta):
    """Write model.npz and model.json to out_dir; meta: res_deg, era5_column, era5_radius_km, origin,
    gap_start, n_train and anything else worth recording."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"z": model.z.numpy(), "post_l": model._post["l"].numpy(), "post_lb": model._post["lb"].numpy(),
              "post_c": model._post["c"].numpy()}
    arrays |= {f"raw_{k}": np.asarray(float(v)) for k, v in model.raw.items()}
    np.savez(out_dir / "model.npz", **arrays)
    info = {"format_version": FORMAT_VERSION, "tool_version": __version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kernel": model.family, "era5": bool(model.era5), "gap_step": bool(model.step),
            "settings": model.cfg, "stats": {k: float(v) for k, v in model.stats.items()},
            "bound": float(model.bound_value), "n_evals": int(model.n_evals),
            "hyperparameters": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in model.hyperparameters().items()}} | meta
    (out_dir / "model.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def load(model_dir):
    """(gp.GP ready to predict, model.json dict) from a folder written by save()."""
    model_dir = Path(model_dir)
    if not (model_dir / "model.json").exists():
        raise FileNotFoundError(f"no model.json in {model_dir}; run clara-gapfill fit first (or pass --model)")
    info = json.loads((model_dir / "model.json").read_text())
    if info.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(f"{model_dir} was written by a newer clara-gapfill (format {info['format_version']})")
    with np.load(model_dir / "model.npz") as a:
        t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=gp.DTYPE)
        raw = {k.removeprefix("raw_"): t(a[k]) for k in a.files if k.startswith("raw_")}
        state = {"family": info["kernel"], "cfg": info["settings"] | {"gap_step": info["gap_step"]},
                 "z": a["z"], "raw": raw, "stats": info["stats"], "era5": info["era5"],
                 "post": {"l": t(a["post_l"]), "lb": t(a["post_lb"]), "c": t(a["post_c"])},
                 "bound_value": info["bound"], "n_evals": info["n_evals"]}
    return gp.GP.from_state(state), info


def import_pt(pt_path, out_dir, res_deg, gap_start):
    """Convert a full_model.pt of the research pipeline (torch.save of GP.state() plus era5_radius_km and
    origin) to model.npz + model.json. The .pt file is unpickled: only use files you trust (your own)."""
    st = torch.load(pt_path, weights_only=False)
    model = gp.GP.from_state(st)
    radius = st.get("era5_radius_km")
    meta = {"res_deg": int(res_deg), "era5_radius_km": None if radius is None else float(radius),
            "era5_column": "era5_night" if radius is None else f"era5_night_r{int(radius)}",
            "origin": st.get("origin", "2020-01-01"), "gap_start": gap_start if model.step else None,
            "imported_from": str(pt_path)}
    return save(model, out_dir, meta)
