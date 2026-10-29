# The model

This page specifies the model that `clara-gapfill fit` estimates and `predict` applies. It is the model
selected by the kernel comparison described at the end. The code is in
[`src/clara_gapfill/gp.py`](../src/clara_gapfill/gp.py).

## Estimand

The estimand is the **monthly-mean CLARA night-side OLR of a grid cell** (radiance, W m⁻² sr⁻¹). Each
observation is the mean of the QC'd CLARA samples whose footprint falls in the cell during the calendar month.
The model time of a cell-month is the middle of the month.

## Observation model

For cell-month *i*:
- *y* is the CLARA cell-month mean;
- *e* is the ERA5 night-time OLR of the same cell and month (flux, W m⁻², smoothed over spherical caps of
  radius *R*);
- *s* is 1 from `time.gap_start` on and 0 before;
- *n* is the number of distinct CLARA overpasses behind *y*.

```math
y_i = \beta_0 + \beta_1 e_i + \beta_2 s_i + f(x_i) + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal{N}\left(0,\ \sigma_0^2 + \frac{\sigma_1^2}{n_i}\right)
```

The terms:
- β₁ (sr⁻¹) carries ERA5's spatial and temporal structure and converts flux into CLARA's radiance units.
- β₂ is the calibration step across the CLARA data gap. The unobserved gap months count as post-gap.
- Samples within one overpass are strongly correlated, so the sampling noise scales with overpasses, not with
  samples.
- σ₀ is the irreducible part of the noise. It collapses to about 0 in the reference fits.

## Inputs

Each cell-month is embedded as a point of R⁶:

```math
x = (u, c, t),\qquad
u = (\cos\varphi\cos\lambda,\ \cos\varphi\sin\lambda,\ \sin\varphi),\qquad
c = (\cos 2\pi\tau,\ \sin 2\pi\tau)
```

- φ and λ are the latitude and longitude of the cell centre.
- τ is the fraction of the year elapsed at the model time (leap-year aware), so December neighbours January.
- *t* is the time in years since `time.origin` (365.25-day years).

The norm of *u* − *u′* is the **chord** between two points. Chords treat the poles and the dateline exactly,
and keep every kernel below positive definite. Great-circle distance would not do that for the Gaussian and
Matérn kernels.

## Covariance

The latent field *f* is a zero-mean GP with two additive components. Component A is space × season (a
pattern that recurs every year); component B is space × long-term time (drift).

```math
k(x, x') = \sigma_A^2\, \kappa(r_A) + \sigma_B^2\, \kappa(r_B)
```

```math
r_A^2 = \frac{\lVert u - u' \rVert^2}{\ell_{sA}^2} + \frac{\lVert c - c' \rVert^2}{\ell_{season}^2},
\qquad
r_B^2 = \frac{\lVert u - u' \rVert^2}{\ell_{sB}^2} + \frac{(t - t')^2}{\ell_{t}^2}
```

The correlation function κ depends on the kernel:

| `model.kernel` | κ(r) |
|---|---|
| `exponential` (selected) | exp(−r) |
| `gaussian` | exp(−r²/2) |
| `matern32` | (1 + √3 r) exp(−√3 r) |

Length scales are fitted as chords. A chord of 2 spans the whole sphere, or half a year on the seasonal
circle. Values above 2 mean "correlated over the whole domain". The hyperparameter tables give the chords,
and the km and day equivalents saturate at 20,015 km and 182.6 days.

## Mean function and standardisation

Inside `fit`, *y* and *e* are standardised with the training rows only; *s* stays 0/1. β, σ and ℓ are
reported in physical units.

The ERA5 predictor is one of the columns `era5_night_r<R>`, the ERA5 night-time mean smoothed over spherical
caps of radius *R* (`era5.smoothing_radii_km`). With `era5.radius_km = "auto"`, *R* is the radius whose
column correlates best with CLARA on the training rows. In cross-validation it is chosen again inside every
fold.

## Sparse approximation (VFE)

The model has *M* inducing inputs *Z*: k-means centres of the training inputs, clustered in coordinates
scaled by the initial length scales and fixed during the fit. Write:
- *K* for the covariance matrices between the training inputs and *Z*;
- Λ for the diagonal noise matrix;
- *m* for the mean function.

The fit maximises the collapsed variational bound of Titsias (2009), generalised to heteroscedastic noise:

```math
\mathcal{L} = \log \mathcal{N}\left(y - m \mid 0,\ Q_{ff} + \Lambda\right)
- \frac{1}{2}\,\mathrm{tr}\left(\Lambda^{-1}\left(K_{ff} - Q_{ff}\right)\right),
\qquad Q_{ff} = K_{fu} K_{uu}^{-1} K_{uf}
```

β and all covariance and noise parameters are fitted jointly, by L-BFGS with a strong-Wolfe line search:
- up to `model.max_iter` iterations per start;
- `model.restarts` starts with the initial length scales ×1, ×0.5 and ×2;
- the best bound wins.

For large tables the bound is accumulated over chunks of `model.chunk_size` rows with gradient
checkpointing, so memory grows with *M* × chunk, not *M* × *N*. With *Z* equal to the training inputs, the
bound is the exact log marginal likelihood; the tests check this.

## Prediction

Let *L* be the Cholesky factor of the inducing-point covariance. Then:

```math
A = L^{-1} K_{uf} \Lambda^{-1/2},\qquad B = I + A A^{\top} = L_B L_B^{\top},\qquad
c = L_B^{-1} A \Lambda^{-1/2} (y - m)
```

At a new cell-month, with `T1` = *L*⁻¹ times the covariance between *Z* and the new point, and `T2` = *L_B*⁻¹
times `T1`:

```math
\mathrm{E}\left[f_\ast\right] = T_2^{\top} c,
\qquad
\mathrm{Var}\left[f_\ast\right] = k_{\ast\ast} - T_1^{\top} T_1 + T_2^{\top} T_2
```

The outputs per cell-month are:
- `pred_mean` = mean function + the latent mean;
- `latent_sd`, the SD of the monthly-mean OLR itself;
- `pred_sd_obs`, which adds the noise of a new cell-month mean from `predict.n_overpasses_for_sd` overpasses.

The model file stores *Z*, *L*, *L_B*, *c*, the parameters and the standardisation, so prediction needs no
training data.

## Global mean

`global_mean_monthly.csv` gives, per month, the area-weighted mean over all cells. Its variance uses the full
posterior covariance of the latent field, not only the variances of the cells:

```math
\mathrm{Var}\left[w^{\top} f_\ast\right] = w^{\top} K_{\ast\ast} w - \lVert T_1 w \rVert^2 + \lVert T_2 w \rVert^2
```

The weights *w* are the cell areas, normalised to sum to 1. Two naive alternatives are given for comparison:
cells treated as independent (too narrow), and the mean of the cell variances.

## How the kernel was selected

The three kernels were compared with blocked leave-one-block-out cross-validation at 5°. The same scheme is
available as `clara-gapfill validate`:

| Scheme | Held-out block | Excluded buffer |
|---|---|---|
| spatial | 60° × 60° blocks partitioning the globe (18 folds) | all cells within 1500 km, in every month |
| temporal | 6-month windows (13 folds; short windows merged) | ±1 month everywhere |
| joint | a 60° block during a 6-month window (18 folds) | within 1500 km and ±1 month |

The block sizes and buffers come from the residual correlation scales of the exploratory analysis. The
selection rule was fixed in advance: lowest **primary score**, the mean over the three schemes of the mean
per-fold RMSE. The exponential kernel scored lowest (39.34 W m⁻² sr⁻¹, against 39.51 Gaussian and 39.74
Matérn 3/2). No pairwise difference was distinguishable from 0 under a paired block bootstrap. The sparse
approximation was checked against the exact GP on a 3000-row subset: the RMSE differed by 0.04. Predictions
were stable for *M* from 250 to 2000.
