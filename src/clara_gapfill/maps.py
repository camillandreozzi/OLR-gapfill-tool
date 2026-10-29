"""Monthly maps of the predictions and the global-mean figure.

Per month: predicted OLR and its uncertainty (latent SD) for every month; observed OLR and residuals for months
CLARA observed. Polar-day cells (ERA5 darkest-hours fallback, extrapolation) are hatched. Colour scales are
shared by every month: OLR from the 1st-99th percentile of predictions and observations, uncertainty up to the
99th percentile, residuals symmetric at the 98th percentile of |residual|.

Colour by job: sequential blue for OLR magnitude, sequential orange for uncertainty, diverging blue <-> red
around a neutral grey for signed residuals; grey cells have no data. Robinson maps with coastlines need
cartopy (pip install "clara-olr-gapfill[maps]"); without it the maps use plain lat-lon axes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import grid  # noqa: E402

try:
    import cartopy.crs as ccrs
except ImportError:  # maps fall back to plain lat-lon axes without coastlines
    ccrs = None

INK, MUTED = "#1f1f1e", "#6b6a66"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NO_DATA = "#d6d5d0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
KERNEL_LABELS = {"exponential": "Exponential", "gaussian": "Gaussian (SE)", "matern32": "Matérn 3/2"}
CLARA_UNIT = "W m$^{-2}$ sr$^{-1}$"
OLR_CMAP = LinearSegmentedColormap.from_list("olr_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]).with_extremes(bad=NO_DATA)
DIVERGING = LinearSegmentedColormap.from_list("div", [
    "#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f4b3b1", "#e34948", "#8f1f1f"]).with_extremes(bad=NO_DATA)
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list("unc", [
    "#fbe3d7", "#f6b496", "#f08a5f", "#eb6834", "#c24d1f", "#8f3412", "#5c1f09"]).with_extremes(bad=NO_DATA)


def style_axes(ax, grid_lines=True):
    ax.tick_params(colors=MUTED, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_color(AXIS)
        spine.set_visible(side in ("left", "bottom"))
    if grid_lines:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def title(ax, text, size=10):
    ax.set_title(text, color=INK, fontsize=size, loc="left")


def map_axes(fig, *args, projection="robinson"):
    """A global map axes (Robinson or PlateCarree with coastlines when cartopy is available)."""
    if ccrs is None:
        ax = fig.add_subplot(*args)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        style_axes(ax, grid_lines=False)
        return ax
    proj = ccrs.Robinson() if projection == "robinson" else ccrs.PlateCarree()
    ax = fig.add_subplot(*args, projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.4, color=INK)
    return ax


def draw_field(ax, values, res, cmap, vmin, vmax):
    """pcolormesh of a (lat, lon) field on the res deg grid."""
    lat_edges, lon_edges = grid.grid_edges(res)
    extra = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
    return ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="flat", **extra)


def hatch(ax, mask, res):
    """Hatch the cells where mask is True (polar-day fallback)."""
    if mask is None or not np.nanmax(mask) > 0:
        return
    lat_c, lon_c = grid.grid_centres(res)
    extra = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
    ax.contourf(lon_c, lat_c, np.nan_to_num(mask), levels=[0.5, 1.5], colors="none", hatches=["////"], **extra)


def colorbar(fig, mesh, ax, label, extend="both", **kw):
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", extend=extend, shrink=kw.pop("shrink", 0.6),
                      aspect=kw.pop("aspect", 35), pad=kw.pop("pad", 0.04), **kw)
    cb.set_label(label, color=INK, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    return cb


def save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def map_one(values, res, cmap, vmin, vmax, label, head, path, mc, extend="both", fallback=None):
    fig = plt.figure(figsize=(9, 5), layout="constrained")
    ax = map_axes(fig, 1, 1, 1, projection=mc["projection"])
    mesh = draw_field(ax, values, res, cmap, vmin, vmax)
    hatch(ax, fallback, res)
    colorbar(fig, mesh, ax, label, extend=extend, shrink=0.7)
    title(ax, head, 10)
    save(fig, path, mc["dpi"])


def residual_maps(ins, oos, res, lim, head, path, mc):
    panels = [(ins, "in-sample (full-data fit)")]
    if oos is not None and np.isfinite(oos).any():
        panels.append((oos, "out-of-sample (spatial block CV)"))
    fig = plt.figure(figsize=(7 * len(panels), 4.6), layout="constrained")
    for k, (v, lab) in enumerate(panels):
        ax = map_axes(fig, 1, len(panels), k + 1, projection=mc["projection"])
        mesh = draw_field(ax, v, res, DIVERGING, -lim, lim)
        ok = np.isfinite(v)
        title(ax, f"{head}\n{lab}: RMSE {np.sqrt(np.nanmean(v ** 2)):.1f} {CLARA_UNIT}, n = {int(ok.sum())}"
              if ok.any() else f"{head}\n{lab}: no data", 9)
    colorbar(fig, mesh, fig.axes, f"residual = observed − predicted ({CLARA_UNIT}); grey = no CLARA observation",
             shrink=0.5)
    save(fig, path, mc["dpi"])


def selected_months(ds, mc):
    times = pd.DatetimeIndex(ds.time.values)
    if mc["months"] == "all":
        return list(range(len(times)))
    if mc["months"] == "observed":
        return [k for k in range(len(times)) if np.nansum(ds.clara_observed.values[k]) > 0]
    wanted = {pd.Period(m, "M") for m in mc["months"]}
    return [k for k, t in enumerate(times) if t.to_period("M") in wanted]


def monthly_maps(g, ds_of_year, cfg, kernel, out_dir):
    """Write the maps of every selected month; g is the full prediction grid (for the shared colour scales)."""
    mc, res = cfg["maps"], cfg["grid"]["res_deg"]
    vals = np.concatenate([g.pred_mean.dropna(), g.clara.dropna()])
    vmin, vmax = np.percentile(vals, [1, 99]).round(-1)
    umax = np.ceil(np.nanpercentile(g.latent_sd, 99))
    res_all = np.r_[g.residual_insample.dropna(), g.residual_oos_spatial.dropna()]
    rlim = np.ceil(np.nanpercentile(np.abs(res_all), 98) / 5) * 5 if len(res_all) else 10.0
    label = KERNEL_LABELS.get(kernel, kernel)
    n_maps = 0
    for year, ds in ds_of_year:
        chosen = selected_months(ds, mc)
        if not chosen:
            continue
        for k in chosen:
            t = pd.Timestamp(ds.time.values[k])
            stem = out_dir / str(year) / f"{t:%Y_%m}"
            n_obs = int(np.nansum(ds.clara_observed.values[k]))
            head = f"{t:%Y-%m} · {label} GP" + ("" if n_obs else " · no CLARA data this month (extrapolation)")
            fb = ds.night_fallback.values[k]
            map_one(ds.pred_mean.values[k], res, OLR_CMAP, vmin, vmax,
                    f"predicted CLARA OLR ({CLARA_UNIT}): posterior mean of the monthly-mean night-side OLR; "
                    "hatched = polar day (extrapolation)", f"{head} · predicted CLARA OLR",
                    stem.with_name(stem.name + "_predicted_olr.png"), mc, fallback=fb)
            map_one(ds.latent_sd.values[k], res, UNCERTAINTY_CMAP, 0, umax,
                    f"posterior SD of the monthly-mean CLARA OLR (latent GP, {CLARA_UNIT}); excludes observation noise; "
                    "hatched = polar day", f"{head} · predictive uncertainty (latent SD)",
                    stem.with_name(stem.name + "_uncertainty.png"), mc, extend="max", fallback=fb)
            n_maps += 2
            if n_obs:
                map_one(ds.clara.values[k], res, OLR_CMAP, vmin, vmax,
                        f"observed CLARA OLR ({CLARA_UNIT}): mean of QC'd night-side samples; grey = not observed",
                        f"{head} · observed CLARA OLR ({n_obs} cells)", stem.with_name(stem.name + "_observed_olr.png"),
                        mc)
                residual_maps(ds.residual_insample.values[k], ds.residual_oos_spatial.values[k], res, rlim,
                              f"{head} · residuals", stem.with_name(stem.name + "_residuals.png"), mc)
                n_maps += 2
        print(f"  {year}: maps written to {out_dir / str(year)}", flush=True)
    return n_maps


def global_mean_figure(gm, kernel, path, dpi):
    label = KERNEL_LABELS.get(kernel, kernel)
    gm = gm.assign(time=pd.to_datetime(gm.time))
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    ax.fill_between(gm.time, gm.global_mean - 2 * gm.sd_full_covariance, gm.global_mean + 2 * gm.sd_full_covariance,
                    color=SERIES[0], alpha=0.2, lw=0, label="±2 SD, full posterior covariance")
    ax.plot(gm.time, gm.global_mean, color=SERIES[0], lw=2, label="predicted global mean (all cells)")
    ax.plot(gm.time, gm.global_mean + 2 * gm.sd_if_independent, color=MUTED, lw=1, ls=":",
            label="+2 SD if cells were independent (too narrow)")
    ax.set_ylabel(f"area-weighted global mean ({CLARA_UNIT})", color=INK)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    title(ax, f"{label} GP: monthly global mean of predicted CLARA OLR (polar-day cells: fallback, extrapolated)")
    save(fig, path, dpi)
