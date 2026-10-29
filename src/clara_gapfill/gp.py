"""Spatio-temporal Gaussian-process model of CLARA OLR (the model of the kernel comparison; exponential won).

Model (identical for every kernel; only kappa changes):

    y_i = beta0 + beta1 * e_i + beta2 * s_i + f(x_i) + eps_i,      eps_i ~ N(0, sigma0^2 + sigma1^2 / n_i)

    y_i   CLARA cell-month mean OLR (radiance, W m-2 sr-1), e_i the night-time ERA5 OLR (flux, W m-2) of the
          cell-month; beta1 (sr-1) also converts ERA5's flux into CLARA's radiance units.
    s_i   post_gap: 1 from time.gap_start on, 0 before (tables.post_gap_indicator); beta2 is the calibration
          step across the CLARA data gap. The term is present when the model's gap_step is true.
    n_i   number of distinct CLARA overpasses behind y_i (samples within one overpass are correlated)
    f     zero-mean GP with the additive covariance

    k(x, x') = sigma_A^2 kappa(r_A) + sigma_B^2 kappa(r_B)
    r_A^2 = |u - u'|^2 / ell_space_A^2 + |c - c'|^2 / ell_season^2     space x season (recurs every year)
    r_B^2 = |u - u'|^2 / ell_space_B^2 + (t - t')^2 / ell_time^2       space x long-term time (drift)

    kappa(r) = exp(-r)                        exponential (the selected kernel)
             = exp(-r^2 / 2)                  gaussian (squared exponential)
             = (1 + sqrt3 r) exp(-sqrt3 r)    matern32

Inputs (embed): u = unit vector of (lat, lon), so |u - u'| is the chord between two points: it treats the
poles and the dateline exactly, and every kappa above stays positive definite (they are isotropic
kernels of Euclidean distance in R^6; with great-circle distance the gaussian and matern32 would not
be). c = (cos 2 pi tau, sin 2 pi tau) with tau the fraction of the year, so December neighbours January.
t = years since the origin. Length scales are reported as chords and, where below 2, in km and days.

Inference: the collapsed variational bound of Titsias (2009) ("VFE" / SGPR) with M inducing inputs Z,
generalised to the heteroscedastic diagonal noise above. Every training observation enters the bound;
with Z equal to the training inputs the bound is the exact log marginal likelihood (tests/test_gp.py).
Z are fixed k-means centres of the training inputs. Hyperparameters and the mean coefficients are found by
maximising the bound (type-II maximum likelihood) with L-BFGS, from fixed data-independent starts.

The response and ERA5 are standardised inside fit() with training statistics only (s stays 0/1, so beta0 is
the pre-gap intercept); everything returned is in physical units.

GP(..., era5=False) drops ERA5 from the mean function (beta1 fixed at 0): the kernel alone then carries the
spatio-temporal structure.
"""

import math

import numpy as np
import pandas as pd
import torch

KERNELS = ("exponential", "gaussian", "matern32")
EARTH_RADIUS_KM = 6371.0
DAYS_PER_YEAR = 365.25
PARAMS = ("sigma_A", "sigma_B", "ell_space_A", "ell_season", "ell_space_B", "ell_time", "sigma0", "sigma1")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def embed(lat, lon, time, origin):
    """(n, 6) inputs [u_x, u_y, u_z, c_x, c_y, t] from lat, lon (deg) and datetime-like time."""
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    time = pd.to_datetime(np.asarray(time))
    year_start = pd.to_datetime(pd.DataFrame({"year": time.year, "month": 1, "day": 1}))
    days_in_year = np.where(time.is_leap_year, 366.0, 365.0)
    tau = (time - pd.DatetimeIndex(year_start)).total_seconds().to_numpy() / 86400 / days_in_year
    t = (time - pd.Timestamp(origin)).total_seconds().to_numpy() / 86400 / DAYS_PER_YEAR
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi),
                            np.cos(2 * np.pi * tau), np.sin(2 * np.pi * tau), t])


def chord_to_km(ell):
    """Chord length on the unit sphere -> great-circle distance in km."""
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(ell) / 2, 0, 1))


def chord_to_days(ell):
    """Chord length on the unit seasonal circle -> separation in days (capped at half a year)."""
    return 2 * np.arcsin(np.clip(np.asarray(ell) / 2, 0, 1)) / (2 * np.pi) * DAYS_PER_YEAR


def km_to_chord(km):
    return 2 * np.sin(np.asarray(km) / EARTH_RADIUS_KM / 2)


def days_to_chord(days):
    return 2 * np.sin(np.asarray(days) / DAYS_PER_YEAR * np.pi)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def kappa(family, r2):
    """Correlation kappa(r) of the family, from squared scaled distances r2."""
    if family == "gaussian":
        return torch.exp(-0.5 * r2)
    r = torch.sqrt(r2.clamp_min(1e-36))
    if family == "exponential":
        return torch.exp(-r)
    if family == "matern32":
        s = math.sqrt(3.0) * r
        return (1 + s) * torch.exp(-s)
    raise ValueError(f"unknown kernel family {family!r}; expected one of {KERNELS}")


def _sqdist(a, b):
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return d.clamp_min(0)


def sq_dists(x1, x2):
    """Squared chord distances in space and season, and squared time differences, between x1 and x2.
    They do not depend on the parameters, so a fit computes them once for its fixed inducing inputs."""
    return (_sqdist(x1[:, :3], x2[:, :3]), _sqdist(x1[:, 3:5], x2[:, 3:5]), (x1[:, 5:6] - x2[:, 5:6].T) ** 2)


def kernel_from_dists(family, d, p):
    """k from the squared distances d = sq_dists(x1, x2)."""
    du, dc, dt = d
    ra = du / p["ell_space_A"] ** 2 + dc / p["ell_season"] ** 2
    rb = du / p["ell_space_B"] ** 2 + dt / p["ell_time"] ** 2
    return p["sigma_A"] ** 2 * kappa(family, ra) + p["sigma_B"] ** 2 * kappa(family, rb)


def kernel(family, x1, x2, p):
    """k(x1, x2) for the additive covariance of the module docstring; p holds positive parameters."""
    return kernel_from_dists(family, sq_dists(x1, x2), p)


def kernel_diag(x, p):
    """diag k(x, x) = sigma_A^2 + sigma_B^2 for every family (kappa(0) = 1)."""
    return torch.full((x.shape[0],), 1.0, dtype=x.dtype) * (p["sigma_A"] ** 2 + p["sigma_B"] ** 2)


def robust_cholesky(k, jitter):
    """Cholesky factor of k + jitter * mean(diag) I, raising the jitter tenfold until it succeeds."""
    scale = k.diagonal().mean().detach()
    eye = torch.eye(k.shape[0], dtype=k.dtype)
    for _ in range(6):
        l, info = torch.linalg.cholesky_ex(k + jitter * scale * eye)
        if int(info) == 0:
            return l
        jitter *= 10
    raise torch.linalg.LinAlgError("Cholesky failed even with large jitter")


# ---------------------------------------------------------------------------
# Inducing inputs
# ---------------------------------------------------------------------------

def inducing_inputs(x, m, cfg_gp, seed):
    """M k-means centres of the training inputs, clustered in coordinates scaled by the initial length scales."""
    from sklearn.cluster import KMeans

    if m >= len(x):
        return x.copy()
    scale = np.array([km_to_chord(cfg_gp["init_ell_space_A_km"])] * 3
                     + [days_to_chord(cfg_gp["init_ell_season_days"])] * 2 + [cfg_gp["init_ell_time_years"]])
    km = KMeans(n_clusters=m, n_init=1, random_state=seed, max_iter=100).fit(x / scale)
    return km.cluster_centers_ * scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GP:
    """Sparse (VFE) GP with the specification of the module docstring for one kernel family."""

    def __init__(self, family, cfg_gp, z=None, era5=True):
        if family not in KERNELS:
            raise ValueError(f"unknown kernel family {family!r}")
        self.family, self.cfg, self.era5 = family, cfg_gp, era5
        self.step = bool(cfg_gp.get("gap_step", False))  # beta2 * post_gap in the mean function
        self.z = None if z is None else torch.as_tensor(z, dtype=DTYPE)
        self.raw = None  # log of positive parameters (standardised units), and beta
        self.stats = None

    # -- parameters ---------------------------------------------------------
    def _init_raw(self, factor=1.0):
        c, sy = self.cfg, self.stats["y_sd"]
        init = {"sigma_A": c["init_sigma_A"] / sy, "sigma_B": c["init_sigma_B"] / sy,
                "ell_space_A": km_to_chord(c["init_ell_space_A_km"]) * factor,
                "ell_season": days_to_chord(c["init_ell_season_days"]) * factor,
                "ell_space_B": km_to_chord(c["init_ell_space_B_km"]) * factor,
                "ell_time": c["init_ell_time_years"] * factor,
                "sigma0": c["init_sigma0"] / sy, "sigma1": c["init_sigma1"] / sy}
        raw = {k: torch.tensor(math.log(float(v)), dtype=DTYPE, requires_grad=True) for k, v in init.items()}
        raw["beta0"] = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
        # without ERA5 beta1 is a fixed 0, not a parameter
        raw["beta1"] = torch.tensor(0.5 if self.era5 else 0.0, dtype=DTYPE, requires_grad=self.era5)
        # without the gap step beta2 is a fixed 0, not a parameter
        raw["beta2"] = torch.tensor(0.0, dtype=DTYPE, requires_grad=self.step)
        return raw

    @staticmethod
    def _pos(raw):
        p = {k: torch.exp(raw[k]) + 1e-6 for k in PARAMS}
        p["beta0"], p["beta1"], p["beta2"] = raw["beta0"], raw["beta1"], raw["beta2"]
        return p

    @staticmethod
    def _mean(p, e, s=None):
        """Mean function beta0 + beta1 e [+ beta2 s] (standardised units)."""
        return p["beta0"] + p["beta1"] * e + (p["beta2"] * s if s is not None else 0.0)

    # -- data ---------------------------------------------------------------
    def _standardise(self, y, e):
        s = self.stats
        return (y - s["y_mean"]) / s["y_sd"], (e - s["e_mean"]) / s["e_sd"]

    @staticmethod
    def _t(a):
        return torch.as_tensor(np.asarray(a, float), dtype=DTYPE)

    def _step(self, s, nrow, training=False):
        """post_gap as a float array when the model has the gap step, else None (s is then ignored). On training
        rows both regimes must be present, otherwise beta2 cannot be separated from beta0."""
        if not self.step:
            return None
        if s is None:
            raise ValueError("the model has the gap step (gap_step = true): pass s = post_gap")
        s = np.asarray(s, float)
        if s.shape != (nrow,):
            raise ValueError(f"post_gap has shape {s.shape}, expected ({nrow},)")
        if training and np.ptp(s) == 0:
            raise ValueError("post_gap is constant on the training rows, so the gap step is not identifiable (a CV "
                             "fold holds out a whole regime): use shorter temporal windows, or gp.gap_step = false")
        return s

    # -- objective ----------------------------------------------------------
    def _terms(self, p, x, y, e, n, d=None, s=None):
        """Return (L, LB, A, c, noise, residual) of the VFE bound in standardised units.
        d = (sq_dists(z, z), sq_dists(z, x)) may be passed to avoid recomputing the distances."""
        noise = p["sigma0"] ** 2 + p["sigma1"] ** 2 / n
        r = y - self._mean(p, e, s)
        d_zz, d_zx = d if d is not None else (sq_dists(self.z, self.z), sq_dists(self.z, x))
        kuu = kernel_from_dists(self.family, d_zz, p)
        l = robust_cholesky(kuu, self.cfg["jitter"])
        kuf = kernel_from_dists(self.family, d_zx, p)
        a = torch.linalg.solve_triangular(l, kuf, upper=False) / torch.sqrt(noise)[None, :]
        b = torch.eye(a.shape[0], dtype=DTYPE) + a @ a.T
        lb = robust_cholesky(b, 1e-12)  # b = I + A A^T has eigenvalues >= 1
        c = torch.linalg.solve_triangular(lb, (a @ (r / torch.sqrt(noise)))[:, None], upper=False)[:, 0]
        return l, lb, a, c, noise, r

    def _chunk_sums(self, p, l, x, y, e, n, s=None):
        """Sums over one chunk of rows: A A^T, A r~, r~.r~, sum log noise, trace term (r~ = r / sqrt(noise))."""
        noise = p["sigma0"] ** 2 + p["sigma1"] ** 2 / n
        rt = (y - self._mean(p, e, s)) / torch.sqrt(noise)
        a = torch.linalg.solve_triangular(l, kernel(self.family, self.z, x, p), upper=False) / torch.sqrt(noise)[None, :]
        return (a @ a.T, a @ rt, (rt * rt).sum(), torch.log(noise).sum(),
                ((p["sigma_A"] ** 2 + p["sigma_B"] ** 2) / noise).sum() - (a * a).sum())

    def _chunked(self, p, x, y, e, n, chunk, s=None):
        """(L, LB, c, logdet, quad, trace) of the VFE bound accumulated over chunks of rows. Each chunk is
        recomputed in the backward pass (gradient checkpointing), so memory is O(M x chunk), not O(M x N)."""
        from torch.utils.checkpoint import checkpoint

        l = robust_cholesky(kernel(self.family, self.z, self.z, p), self.cfg["jitter"])
        keys = sorted(p)
        sums = None
        for i in range(0, len(y), chunk):
            sl = slice(i, i + chunk)
            fn = lambda l_, *vals, sl=sl: self._chunk_sums(dict(zip(keys, vals)), l_, x[sl], y[sl], e[sl], n[sl],
                                                           None if s is None else s[sl])
            part = checkpoint(fn, l, *[p[k] for k in keys], use_reentrant=False) if torch.is_grad_enabled() \
                else fn(l, *[p[k] for k in keys])
            sums = part if sums is None else tuple(s + q for s, q in zip(sums, part))
        aat, ar, rr, lognoise, trace = sums
        lb = robust_cholesky(torch.eye(aat.shape[0], dtype=DTYPE) + aat, 1e-12)
        c = torch.linalg.solve_triangular(lb, ar[:, None], upper=False)[:, 0]
        logdet = 2 * torch.log(lb.diagonal()).sum() + lognoise
        return l, lb, c, logdet, rr - (c * c).sum(), trace

    def bound_chunked(self, p, x, y, e, n, chunk, s=None):
        """The same Titsias bound as bound(), accumulated over chunks of `chunk` rows."""
        _, _, _, logdet, quad, trace = self._chunked(p, x, y, e, n, chunk, s)
        return -0.5 * (y.shape[0] * math.log(2 * math.pi) + logdet + quad + trace)

    def bound(self, p, x, y, e, n, d=None, s=None):
        """Titsias collapsed lower bound on log p(y), standardised units."""
        l, lb, a, c, noise, r = self._terms(p, x, y, e, n, d, s)
        nobs = y.shape[0]
        logdet = 2 * torch.log(lb.diagonal()).sum() + torch.log(noise).sum()
        quad = (r * r / noise).sum() - (c * c).sum()
        trace = (kernel_diag(x, p) / noise).sum() - (a * a).sum()
        return -0.5 * (nobs * math.log(2 * math.pi) + logdet + quad + trace)

    # -- fitting ------------------------------------------------------------
    def fit(self, x, y, e, n, verbose=False, s=None):
        """Fit on training rows: inputs x (n, 6), CLARA y, ERA5 e, overpasses n (all physical units), and
        s = post_gap (0/1), required when the model has the gap step."""
        y, n = np.asarray(y, float), np.asarray(n, float)
        s = self._step(s, len(y), training=True)
        e = np.asarray(e, float) if self.era5 else np.zeros_like(y)  # without ERA5, e is never read
        self.stats = {"y_mean": y.mean(), "y_sd": y.std(), "e_mean": e.mean(), "e_sd": e.std() if self.era5 else 1.0}
        if self.z is None:
            self.z = torch.as_tensor(inducing_inputs(x, self.cfg["n_inducing"], self.cfg, self.cfg["seed"]),
                                     dtype=DTYPE)
        xt, nt = self._t(x), self._t(n)
        yt, et = (self._t(v) for v in self._standardise(y, e))
        st = None if s is None else self._t(s)
        chunk = self.cfg.get("chunk_size", 0)
        if chunk and len(y) > chunk:  # large data: chunked, checkpointed bound, memory O(M x chunk)
            d = None
            objective = lambda p: self.bound_chunked(p, xt, yt, et, nt, chunk, st)
        else:  # cache the parameter-free distances, memory O(M x N)
            d = (sq_dists(self.z, self.z), sq_dists(self.z, xt))
            objective = lambda p: self.bound(p, xt, yt, et, nt, d, st)
        best = None
        factors = [1.0, 0.5, 2.0][: self.cfg["restarts"]]
        for factor in factors:
            raw = self._init_raw(factor)
            opt = torch.optim.LBFGS([v for v in raw.values() if v.requires_grad], lr=1.0, max_iter=self.cfg["max_iter"], history_size=20,
                                    line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
            n_eval = [0]

            def closure():
                opt.zero_grad()
                loss = -objective(self._pos(raw)) / len(y)
                loss.backward()
                n_eval[0] += 1
                if verbose and n_eval[0] % 10 == 0:
                    print(f"    restart x{factor}: {n_eval[0]} evaluations, bound / N {-float(loss.detach()):.5f}", flush=True)
                return loss

            try:
                opt.step(closure)
                with torch.no_grad():
                    value = float(objective(self._pos(raw)))
            except (torch.linalg.LinAlgError, RuntimeError) as err:
                value = -np.inf
                if verbose:
                    print(f"  restart x{factor}: failed ({err})")
            if verbose:
                print(f"  {self.family} restart x{factor}: bound {value:.2f} after {n_eval[0]} evaluations")
            if np.isfinite(value) and (best is None or value > best[0]):
                best = (value, {k: v.detach().clone() for k, v in raw.items()}, n_eval[0])
        if best is None:
            raise RuntimeError(f"{self.family}: every restart failed")
        self.bound_value, self.raw, self.n_evals = best
        with torch.no_grad():
            p = self._pos(self.raw)
            if d is None:
                l, lb, c, _, _, _ = self._chunked(p, xt, yt, et, nt, chunk, st)
            else:
                l, lb, _, c, _, _ = self._terms(p, xt, yt, et, nt, d, st)
        self._post = {"p": p, "l": l, "lb": lb, "c": c}
        del d
        return self

    # -- prediction ---------------------------------------------------------
    def _mean_fn(self, e, s=None):
        """Mean function beta0 + beta1 e [+ beta2 s] in physical units (CLARA's, W m-2 sr-1)."""
        b0, b1, b2 = self.coefficients()
        return b0 + b1 * np.asarray(e, float) + (b2 * s if s is not None else 0.0)

    def coefficients(self):
        """(beta0, beta1, beta2) of the mean function in physical units:
        CLARA = beta0 + beta1 * ERA5 + beta2 * post_gap (beta2 = 0 without the gap step)."""
        s, p = self.stats, self._post["p"]
        b1 = float(p["beta1"]) * s["y_sd"] / s["e_sd"]
        b0 = s["y_mean"] + float(p["beta0"]) * s["y_sd"] - b1 * s["e_mean"]
        return b0, b1, float(p["beta2"]) * s["y_sd"]

    def predict(self, x, e, n=None, batch=20000, s=None):
        """Predictions at inputs x with ERA5 e and post_gap s (required with the gap step). Returns a dict of
        numpy arrays in physical units: mean (latent monthly-mean CLARA OLR), mean_fn (beta0 + beta1 e
        [+ beta2 s]), latent_var, and, when n (overpasses) is given, obs_var = latent_var + noise for a new
        cell-month mean from n overpasses."""
        post, sy = self._post, self.stats["y_sd"]
        s = self._step(s, len(x))
        if not self.era5:
            e = np.zeros(len(x))
        out = {"mean": [], "latent_var": []}
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xs = self._t(x[i:i + batch])
                kus = kernel(self.family, self.z, xs, post["p"])
                t1 = torch.linalg.solve_triangular(post["l"], kus, upper=False)
                t2 = torch.linalg.solve_triangular(post["lb"], t1, upper=False)
                out["mean"].append((t2.T @ post["c"]).numpy())
                var = kernel_diag(xs, post["p"]) - (t1 * t1).sum(0) + (t2 * t2).sum(0)
                out["latent_var"].append(var.clamp_min(0).numpy())
        mean_fn = self._mean_fn(e, s)
        res = {"mean_fn": mean_fn, "mean": mean_fn + np.concatenate(out["mean"]) * sy,
               "latent_var": np.concatenate(out["latent_var"]) * sy ** 2}
        if n is not None:
            p = post["p"]
            noise = (float(p["sigma0"]) ** 2 + float(p["sigma1"]) ** 2 / np.asarray(n, float)) * sy ** 2
            res["obs_var"] = res["latent_var"] + noise
        return res

    def prior_quadratic(self, x, weights, block=1000):
        """w^T K(x, x) w (prior covariance of the weighted average; weights normalised to sum 1),
        computed in row blocks so the n* x n* matrix is never held at once."""
        w = np.asarray(weights, float)
        w = w / w.sum()
        with torch.no_grad():
            xs, wt = self._t(x), self._t(w)
            return float(sum(wt[i:i + block] @ (kernel(self.family, xs[i:i + block], xs, self._post["p"]) @ wt)
                             for i in range(0, len(xs), block)))

    def aggregate(self, x, e, weights, prior_quad=None, s=None):
        """Mean and variance of the weighted average sum_i w_i f*(x_i) (weights normalised to sum 1),
        using the full posterior covariance of the latent values rather than their variances alone.
        prior_quad = prior_quadratic(x, weights) may be passed when it is already known; s = post_gap."""
        w = np.asarray(weights, float)
        w = w / w.sum()
        post, sy = self._post, self.stats["y_sd"]
        s = self._step(s, len(x))
        if not self.era5:
            e = np.zeros(len(x))
        if prior_quad is None:
            prior_quad = self.prior_quadratic(x, w)
        with torch.no_grad():
            xs, wt = self._t(x), self._t(w)
            kus = kernel(self.family, self.z, xs, post["p"])
            t1 = torch.linalg.solve_triangular(post["l"], kus, upper=False)
            t2 = torch.linalg.solve_triangular(post["lb"], t1, upper=False)
            var = prior_quad - ((t1 @ wt) ** 2).sum() + ((t2 @ wt) ** 2).sum()
            mean = wt @ (t2.T @ post["c"])
        return float(w @ self._mean_fn(e, s) + float(mean) * sy), float(var.clamp_min(0)) * sy ** 2

    def state(self):
        """Everything predict() needs, for torch.save (training data are not stored)."""
        post = {k: self._post[k] for k in ("l", "lb", "c")}
        return {"family": self.family, "cfg": self.cfg, "z": self.z, "raw": self.raw, "stats": self.stats,
                "post": post, "bound_value": self.bound_value, "n_evals": self.n_evals, "era5": self.era5}

    @classmethod
    def from_state(cls, st):
        model = cls(st["family"], st["cfg"], z=st["z"], era5=st.get("era5", True))  # older states: with ERA5
        model.raw, model.stats = st["raw"], st["stats"]
        model.bound_value, model.n_evals = st["bound_value"], st["n_evals"]
        model._post = {"p": cls._pos(st["raw"]), **st["post"]}
        return model

    def hyperparameters(self):
        """Fitted parameters in physical units, with their interpretation-ready names.

        The *_wm2 values are in CLARA's units, W m-2 sr-1 (beta1 in sr-1); the suffix is kept so the names match
        the saved hyperparameter tables."""
        p, sy = self._post["p"], self.stats["y_sd"]
        b0, b1, b2 = self.coefficients()
        return {
            "kernel": self.family, "bound": self.bound_value, "n_evals": self.n_evals, "n_inducing": len(self.z),
            "beta0_wm2": b0, "beta1": b1, "beta2_wm2": b2,
            "sigma_A_wm2": float(p["sigma_A"]) * sy, "ell_space_A_km": float(chord_to_km(float(p["ell_space_A"]))),
            "ell_space_A_chord": float(p["ell_space_A"]),
            "ell_season_days": float(chord_to_days(float(p["ell_season"]))), "ell_season_chord": float(p["ell_season"]),
            "sigma_B_wm2": float(p["sigma_B"]) * sy, "ell_space_B_km": float(chord_to_km(float(p["ell_space_B"]))),
            "ell_space_B_chord": float(p["ell_space_B"]), "ell_time_years": float(p["ell_time"]),
            "sigma0_wm2": float(p["sigma0"]) * sy, "sigma1_wm2": float(p["sigma1"]) * sy,
        }


def exact_log_marginal(family, p, x, y, e, n, s=None):
    """Exact log N(y | beta0 + beta1 e [+ beta2 s], K + noise) in the units of y (for tests and benchmarks)."""
    noise = p["sigma0"] ** 2 + p["sigma1"] ** 2 / n
    r = y - GP._mean(p, e, s)
    k = kernel(family, x, x, p) + torch.diag(noise)
    l = torch.linalg.cholesky(k)
    alpha = torch.cholesky_solve(r[:, None], l)[:, 0]
    return -0.5 * (len(y) * math.log(2 * math.pi) + 2 * torch.log(l.diagonal()).sum() + r @ alpha)


def exact_predict(family, p, x, y, e, n, xs, es, s=None, ss=None):
    """Exact GP latent mean and variance at xs (units of y), for tests and benchmarks."""
    noise = p["sigma0"] ** 2 + p["sigma1"] ** 2 / n
    r = y - GP._mean(p, e, s)
    l = torch.linalg.cholesky(kernel(family, x, x, p) + torch.diag(noise))
    ks = kernel(family, x, xs, p)
    alpha = torch.cholesky_solve(r[:, None], l)[:, 0]
    v = torch.linalg.solve_triangular(l, ks, upper=False)
    return GP._mean(p, es, ss) + ks.T @ alpha, kernel_diag(xs, p) - (v * v).sum(0)
