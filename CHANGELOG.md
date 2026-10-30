# Changelog

## 0.1.0 (unreleased)

First release.
- The full pipeline from raw CLARA (IDL `.save` + geolocation CSV) and ERA5 (monthly means by hour of day) to
  gap-filled monthly maps: `prepare`, `fit`, `predict`, `maps` and `run`, with stage manifests so that only
  what changed is redone.
- The selected model: an exponential-kernel sparse GP with an ERA5 mean function and a post-gap step. The
  default settings are those of the reference fit (1000 inducing points, 3 starts).
- All settings, including the grid resolution (any divisor of 180°), are in one TOML config, with
  `--set` overrides.
- Shortcuts: `fit --table` and `predict --grid` for your own tables, and portable model files
  (`model.npz` + `model.json`, no pickle). `import-model` converts models of the research pipeline.
- `validate`: blocked spatial, temporal and joint cross-validation.
- `download-era5` (CDS) and `demo` (synthetic data).
- At 5°, preprocessing and prediction were checked against the original research pipeline. The training
  table and the prediction grid are identical; the predictions of the imported reference model are identical
  to 3 decimals.
- A refit with the default settings reaches the reference bound to 2·10⁻⁵ per observation. It differs only
  because k-means places the inducing points slightly differently across machines. Its maps differ from the
  reference by 0.18 W m⁻² sr⁻¹ RMS.
