"""
aifs.plot
=========
Cartopy-based helpers for visualising AIFS forecast output on a global map.

All functions return ``matplotlib.figure.Figure`` objects so they work in
both notebooks (``plt.show()``) and scripts (``fig.savefig(...)``).

Quickstart
----------
    from aifs.plot import plot_field, plot_field_sequence

    # Single map
    fig = plot_field(state, "2t", title="2-m Temperature — T+6h")
    fig.savefig("t2m_T+6.png", dpi=150)

    # Multi-panel sequence
    fig = plot_field_sequence(states, "2t", max_steps=4)
    fig.savefig("t2m_sequence.png", dpi=150)
"""

from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Variable metadata ─────────────────────────────────────────────────────────

#: Variables that can be extracted from forecast state dicts
PLOTTABLE = [
    "2t", "msl", "sp", "tcw", "10u", "10v", "swh", "mwp",
    "t_850", "t_500", "u_850", "v_850", "z_500", "q_700",
]

_CMAP = {
    "2t":    "RdBu_r", "t_850": "RdBu_r", "t_500": "RdBu_r",
    "msl":   "viridis", "sp":    "viridis",
    "10u":   "RdBu",    "10v":   "RdBu",
    "u_850": "RdBu",    "v_850": "RdBu",
    "swh":   "Blues",   "mwp":   "Blues",   "tcw":   "Blues",
    "z_500": "plasma",  "q_700": "YlGn",
}

_UNITS = {
    "2t":    "K",      "t_850": "K",      "t_500": "K",
    "msl":   "Pa",     "sp":    "Pa",     "z_500": "m²/s²",
    "10u":   "m/s",    "10v":   "m/s",    "u_850": "m/s",    "v_850": "m/s",
    "swh":   "m",      "mwp":   "s",      "tcw":   "kg/m²",  "q_700": "kg/kg",
}

_LONG_NAME = {
    "2t":    "2-m Temperature",
    "msl":   "Mean Sea-Level Pressure",
    "sp":    "Surface Pressure",
    "tcw":   "Total Column Water",
    "10u":   "10-m U Wind",
    "10v":   "10-m V Wind",
    "swh":   "Significant Wave Height",
    "mwp":   "Mean Wave Period",
    "t_850": "Temperature at 850 hPa",
    "t_500": "Temperature at 500 hPa",
    "u_850": "U Wind at 850 hPa",
    "v_850": "V Wind at 850 hPa",
    "z_500": "Geopotential at 500 hPa",
    "q_700": "Specific Humidity at 700 hPa",
}


# ── Grid coordinate extraction ────────────────────────────────────────────────

def _get_latlons(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lats, lons) for the grid the forecast was run on.

    The anemoi tensor handler injects ``state["latitudes"]`` and
    ``state["longitudes"]`` from the checkpoint metadata before the first
    inference step, and these are propagated to every output state via
    ``new_states = input_states.copy()``.  We read them directly — no
    separate grid-geometry lookup needed.

    Longitudes are returned in the range [0, 360) as stored by anemoi;
    callers that need [-180, 180) should call ``_to_180(lons)``.
    """
    lats = state.get("latitudes")
    lons = state.get("longitudes")

    if lats is None or lons is None:
        raise KeyError(
            "State dict does not contain 'latitudes'/'longitudes'. "
            "Make sure you are passing a state returned by run_forecast() "
            "and have not stripped those keys."
        )

    lats = np.asarray(lats).ravel()
    lons = np.asarray(lons).ravel()

    if len(lats) < 3 or len(lons) < 3:
        raise ValueError(
            f"Grid has only {len(lats)} points — expected ~542 080 for N320. "
            "The state latitudes/longitudes may be corrupt."
        )

    return lats, lons


def _to_180(lons: np.ndarray) -> np.ndarray:
    """Normalise longitudes from [0, 360) to [-180, 180) for Cartopy."""
    return np.where(lons > 180, lons - 360, lons)


def _extract_field(state: dict, variable: str) -> np.ndarray | None:
    """Pull ``variable`` out of ``state["fields"]``, return None if missing."""
    return state.get("fields", {}).get(variable)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_field(
    state: dict,
    variable: str,
    title: str | None = None,
    projection: str = "Robinson",
    figsize: tuple[float, float] = (14, 7),
    vmin: float | None = None,
    vmax: float | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot a single forecast field on a global map.

    Parameters
    ----------
    state:
        One element from the list returned by :func:`aifs.forecast.run_forecast`.
    variable:
        Short name of the field to plot (e.g. ``"2t"``, ``"msl"``).
        See :data:`PLOTTABLE` for supported names.
    title:
        Figure title; defaults to ``"{long_name} — {state_date}"``.
    projection:
        Cartopy projection class name (e.g. ``"Robinson"``, ``"PlateCarree"``).
    figsize:
        Matplotlib figure size in inches.
    vmin, vmax:
        Colour-scale limits; auto-derived from percentiles if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.tri as tri

    data = _extract_field(state, variable)
    if data is None:
        raise KeyError(
            f"Variable '{variable}' not found in forecast state. "
            f"Available: {sorted(state.get('fields', {}).keys())}"
        )

    data     = np.asarray(data).ravel()
    lats, lons = _get_latlons(state)
    lons_plot  = _to_180(lons)

    units = _UNITS.get(variable, "")
    lname = _LONG_NAME.get(variable, variable)
    dt    = state.get("date", "")

    # Default colormap: coolwarm gives blue=cold, red=warm; fall back per variable
    cmap = _CMAP.get(variable, "coolwarm")

    if vmin is None:
        vmin = float(np.nanpercentile(data, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data, 98))

    proj_cls = getattr(ccrs, projection, ccrs.Robinson)
    proj     = proj_cls()
    pc       = ccrs.PlateCarree()

    # Pre-project coordinates into projection space before triangulating
    xy    = proj.transform_points(pc, lons_plot, lats)   # (N, 3)
    x, y  = xy[:, 0], xy[:, 1]

    # Drop points that failed to project
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, data = x[valid], y[valid], data[valid]

    triangulation = tri.Triangulation(x, y)

    # Mask triangles spanning the antimeridian (threshold in projection metres)
    x_verts    = x[triangulation.triangles]
    max_x_span = np.max(x_verts, axis=1) - np.min(x_verts, axis=1)
    triangulation.set_mask(max_x_span > 1e6)

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.5)

    # No transform= — coordinates are already in projection space
    pcm = ax.tripcolor(
        triangulation, data,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="gouraud",
    )

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                        pad=0.04, fraction=0.03, shrink=0.8)
    cbar.set_label(f"{lname}  [{units}]", fontsize=10)

    if title is None:
        title = f"{lname}  —  {dt}"
    ax.set_title(title, fontsize=12, pad=10)

    fig.tight_layout()

    return fig


def plot_field_sequence(
    states: list[dict],
    variable: str,
    max_steps: int = 4,
    figsize_per_panel: tuple[float, float] = (7, 3.5),
    projection: str = "Robinson",
    shared_colorscale: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Plot a sequence of forecast steps side-by-side in a single figure.

    Parameters
    ----------
    states:
        List of state dicts from :func:`aifs.forecast.run_forecast`.
    variable:
        Variable short name.
    max_steps:
        Maximum number of panels (steps) to show.
    figsize_per_panel:
        Width × height for each panel in inches.
    projection:
        Cartopy projection class name.
    shared_colorscale:
        Use the same colour limits for all panels (recommended).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.tri as tri

    steps = states[:max_steps]
    n     = len(steps)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig_w    = figsize_per_panel[0] * ncols
    fig_h    = figsize_per_panel[1] * nrows
    proj_cls = getattr(ccrs, projection, ccrs.Robinson)
    proj     = proj_cls()
    pc       = ccrs.PlateCarree()

    # Pre-project coordinates once from the first state
    lats, lons = _get_latlons(steps[0])
    lons_plot  = _to_180(lons)

    xy    = proj.transform_points(pc, lons_plot, lats)
    x, y  = xy[:, 0], xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]

    triangulation = tri.Triangulation(x, y)
    x_verts       = x[triangulation.triangles]
    max_x_span    = np.max(x_verts, axis=1) - np.min(x_verts, axis=1)
    triangulation.set_mask(max_x_span > 1e6)

    cmap  = _CMAP.get(variable, "coolwarm")
    units = _UNITS.get(variable, "")
    lname = _LONG_NAME.get(variable, variable)

    if shared_colorscale:
        all_data = np.concatenate(
            [np.asarray(_extract_field(s, variable)).ravel()[valid]
             for s in steps if _extract_field(s, variable) is not None]
        )
        vmin = float(np.nanpercentile(all_data, 2))
        vmax = float(np.nanpercentile(all_data, 98))
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": proj},
    )
    axes_flat = np.array(axes).ravel()

    for idx, (state, ax) in enumerate(zip(steps, axes_flat)):
        data = _extract_field(state, variable)
        if data is None:
            ax.set_visible(False)
            continue

        data  = np.asarray(data).ravel()[valid]   # apply same valid mask
        _vmin = vmin if shared_colorscale else float(np.nanpercentile(data, 2))
        _vmax = vmax if shared_colorscale else float(np.nanpercentile(data, 98))

        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

        # No transform= — coordinates already in projection space
        pcm = ax.tripcolor(
            triangulation, data,
            cmap=cmap, vmin=_vmin, vmax=_vmax,
            shading="gouraud",
        )
        step_label = f"T+{(idx + 1) * 6}h  ({state.get('date', '')})"
        ax.set_title(step_label, fontsize=9)

        fig.colorbar(pcm, ax=ax, orientation="horizontal",
                     pad=0.04, fraction=0.04, shrink=0.85,
                     label=f"{units}")

    for ax in axes_flat[len(steps):]:
        ax.set_visible(False)

    fig.suptitle(f"{lname} — AIFS Forecast", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig