"""Command-line interface: clara-gapfill <command> [-c config.toml] [--set section.key=value ...]."""

import argparse
import os
import sys
from pathlib import Path

from . import __version__

EPILOG = """typical use:
  clara-gapfill init-config my_config.toml      # then edit paths, grid.res_deg, time range ...
  clara-gapfill download-era5 -c my_config.toml # ERA5 by-hour ttr from the Copernicus CDS (~0.4 GB per year)
  clara-gapfill run -c my_config.toml           # prepare -> fit -> predict -> maps
  clara-gapfill run -c my_config.toml --set grid.res_deg=1   # the same at 1 deg, in output/res1/

shortcuts:
  clara-gapfill fit -c my_config.toml --table my_cell_months.csv
  clara-gapfill predict -c my_config.toml --model output/res5/exponential/model --grid my_grid.csv
  clara-gapfill demo                            # synthetic data, checks the installation
"""


def _common(p):
    p.add_argument("-c", "--config", help="TOML file with your settings (defaults for anything left out)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one setting, e.g. --set grid.res_deg=1 (repeatable)")
    p.add_argument("--force", action="store_true", help="re-run the stage(s) even when up to date")
    p.add_argument("-q", "--quiet", action="store_true", help="less output while fitting")


def parser():
    ap = argparse.ArgumentParser(prog="clara-gapfill", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Gap-filled monthly maps of CLARA (NorSat-1) night-side OLR from ERA5 "
                                             "with a sparse spatio-temporal Gaussian process.")
    ap.add_argument("--version", action="version", version=f"clara-gapfill {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("init-config", help="write a commented config file with every setting")
    p.add_argument("path", nargs="?", default="clara_gapfill.toml")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("show-config", help="print the resolved settings")
    _common(p)

    p = sub.add_parser("download-era5", help="download the ERA5 by-hour ttr product for the time range")
    _common(p)
    p.add_argument("--dry-run", action="store_true", help="print the CDS requests only")

    p = sub.add_parser("prepare", help="raw CLARA + ERA5 -> training table and prediction grid")
    _common(p)
    p.add_argument("--only", choices=["clara", "era5", "tables"], help="run one preparation step")

    p = sub.add_parser("fit", help="fit the GP on the training table")
    _common(p)
    p.add_argument("--table", help="your own cell-month table (CSV or Parquet) instead of the prepared one")

    p = sub.add_parser("predict", help="predict every cell of every month")
    _common(p)
    p.add_argument("--model", help="model folder (model.json + model.npz); default: this config's fit")
    p.add_argument("--grid", help="your own prediction grid (CSV or Parquet) instead of the prepared one")
    p.add_argument("--out", help="output folder (default: output/res<N>/<kernel>/predictions)")

    p = sub.add_parser("maps", help="monthly maps of the predictions")
    _common(p)
    p.add_argument("--predictions", help="predictions folder (default: this config's)")

    p = sub.add_parser("validate", help="blocked spatial / temporal / joint cross-validation (slow)")
    _common(p)
    p.add_argument("--table", help="your own cell-month table instead of the prepared one")
    p.add_argument("--schemes", nargs="+", choices=["spatial", "temporal", "joint"])
    p.add_argument("--folds", nargs="+", type=int, help="only these fold numbers (a quick look)")

    p = sub.add_parser("run", help="prepare, fit, predict and map, skipping what is up to date")
    _common(p)

    p = sub.add_parser("demo", help="end-to-end run on synthetic data (no downloads)")
    p.add_argument("--out", default="clara_gapfill_demo", help="folder for the demo files")
    p.add_argument("--res", type=int, default=10, help="grid resolution of the demo (deg)")
    p.add_argument("--quick", action="store_true", help="tiny fit and three maps (seconds)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")

    p = sub.add_parser("import-model", help="convert a full_model.pt of the research pipeline (trusted files only)")
    p.add_argument("pt", help="full_model.pt")
    p.add_argument("--out", required=True, help="model folder to write")
    p.add_argument("--res", type=int, default=5, help="grid resolution the model was fitted at (deg)")
    p.add_argument("--gap-start", default="2021-09", help="time.gap_start the model was fitted with")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    from . import config

    try:
        if args.command == "init-config":
            path = Path(args.path)
            if path.exists() and not args.overwrite:
                sys.exit(f"{path} exists; use --overwrite")
            path.write_text(config.default_text())
            print(f"wrote {path}; edit it, then: clara-gapfill run -c {path}")
            return
        if args.command == "demo":
            from . import demo
            demo.run(Path(args.out), args.res, args.quick, args.set)
            return
        if args.command == "import-model":
            from . import model_io
            info = model_io.import_pt(args.pt, args.out, args.res, args.gap_start)
            print(f"wrote {args.out}/model.npz and model.json ({info['kernel']}, {info['res_deg']} deg, "
                  f"ERA5 column {info['era5_column']})")
            return

        cfg = config.load(args.config, args.set)
        if args.command == "show-config":
            import json
            print(json.dumps(cfg, indent=2, default=str))
            return
        if args.command == "download-era5":
            from . import era5
            era5.download(list(config.months(cfg)), cfg["paths"]["era5_dir"], args.dry_run)
            return

        from . import pipeline
        verbose = not args.quiet
        if args.command == "prepare":
            steps = [args.only] if args.only else ["clara", "era5", "tables"]
            for step in steps:
                getattr(pipeline, f"stage_{step}")(cfg, args.force)
        elif args.command == "fit":
            pipeline.stage_fit(cfg, args.force, args.table, verbose)
        elif args.command == "predict":
            pipeline.stage_predict(cfg, args.force, args.model, args.grid, args.out)
        elif args.command == "maps":
            pipeline.stage_maps(cfg, args.force, args.predictions)
        elif args.command == "validate":
            pipeline.stage_validate(cfg, args.table, args.schemes, args.folds, verbose)
        elif args.command == "run":
            pipeline.run_all(cfg, args.force, verbose)
    except (config.ConfigError, FileNotFoundError, ValueError) as err:
        if os.environ.get("CLARA_GAPFILL_DEBUG"):
            raise
        sys.exit(f"clara-gapfill {args.command}: {err}\n(set CLARA_GAPFILL_DEBUG=1 for the full traceback)")


if __name__ == "__main__":
    main()
