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

    units = _UNITS.get(variable, "")
    lname = _LONG_NAME.get(variable, variable)
    dt = state.get("date", "")

    lats, lons = _get_latlons(state)
    lons_plot  = _to_180(lons)

    fig, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    triangulation = tri.Triangulation(lons_plot, lats)

    contour = ax.tricontourf(triangulation, data, levels=20, transform=ccrs.PlateCarree(), cmap="RdBu_r")
    cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.7, label=variable)
    cbar.set_label(f"{lname}  [{units}]", fontsize=10)

    plt.title(variable .format(dt))
    fig.tight_layout()
    plt.show()

    return fig

