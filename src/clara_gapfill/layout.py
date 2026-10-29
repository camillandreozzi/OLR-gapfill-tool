"""Where every file of a run lives.

    <work_dir>/extracted/<archive>/             extracted CLARA archives (never written into clara_dir)
    <work_dir>/clara_geolocated.parquet         every CLARA sample with its interpolated footprint geometry
    <work_dir>/era5_night_1deg.nc               ERA5 monthly night-time OLR on 1 deg cells
    <work_dir>/res<N>/training_table.parquet    one row per CLARA-observed cell-month (the model's training data)
    <work_dir>/res<N>/prediction_grid.parquet   every cell of every month, with the ERA5 predictors
    <output_dir>/res<N>/preprocessing/          QC counts, validation checks, observation summary
    <output_dir>/res<N>/<kernel>/model/         fitted model (model.npz + model.json), hyperparameters, in-sample fit
    <output_dir>/res<N>/<kernel>/predictions/   grid_predictions table, NetCDF per year, global mean
    <output_dir>/res<N>/<kernel>/maps/<yyyy>/   monthly maps
    <output_dir>/res<N>/<kernel>/validation/    blocked cross-validation
"""

from pathlib import Path


class Layout:
    def __init__(self, cfg):
        self.work = Path(cfg["paths"]["work_dir"])
        self.out = Path(cfg["paths"]["output_dir"])
        self.res = cfg["grid"]["res_deg"]
        self.kernel = cfg["model"]["kernel"]
        self.extracted = self.work / "extracted"
        self.geolocated = self.work / "clara_geolocated.parquet"
        self.era5_night = self.work / "era5_night_1deg.nc"
        self.work_res = self.work / f"res{self.res}"
        self.training_table = self.work_res / "training_table.parquet"
        self.prediction_grid = self.work_res / "prediction_grid.parquet"
        self.out_res = self.out / f"res{self.res}"
        self.preprocessing = self.out_res / "preprocessing"
        self.run = self.out_res / self.kernel
        self.model = self.run / "model"
        self.predictions = self.run / "predictions"
        self.maps = self.run / "maps"
        self.validation = self.run / "validation"

    def grid_table(self, fmt):
        return self.predictions / f"grid_predictions.{fmt}"
