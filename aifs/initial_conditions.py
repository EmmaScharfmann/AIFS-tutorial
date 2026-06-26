import datetime
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Meteorological variable lists ─────────────────────────────────────────────

#: Surface parameters (levtype=sfc)
PARAM_SFC = [
    "10u", "10v", "2d", "2t", "msl", "skt", "sp",
    "tcw", "lsm", "z", "slor", "sdor", "sd",
]

#: Soil parameters (levtype=sfc, levelist=[1,2])
PARAM_SOIL = ["vsw", "sot"]
SOIL_LEVELS = [1, 2]

#: Ocean-wave parameters (stream=wave)
PARAM_WAVE = [
    "wmb", "h1012", "h1214", "h1417", "h1721",
    "h2125", "h2530", "mwd", "cdww", "mwp", "swh",
]

#: Pressure-level parameters
PARAM_PL = ["gh", "t", "u", "v", "q"]
LEVELS   = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]

SOURCE = "ecmwf"

# ── Cache helpers ─────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path("ic_cache")


def _cache_path(date: datetime.datetime, cache_dir: Path) -> Path:
    return cache_dir / f"ic_{date.strftime('%Y%m%dT%H%M%S')}.npz"


def _save(date: datetime.datetime, fields: dict, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(date, cache_dir)
    np.savez_compressed(str(path), **fields)
    return path


def _try_load(date: datetime.datetime, cache_dir: Path):
    """Return ``(fields_dict, path)`` if cached, else ``(None, None)``."""
    path = _cache_path(date, cache_dir)
    if path.exists():
        return dict(np.load(str(path))), path
    return None, None


def list_cached(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[Path]:
    """Return all cached .npz files, newest first."""
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("ic_*.npz"), reverse=True)


# ── Download helpers ──────────────────────────────────────────────────────────

def _fetch_fields(ekd, ekr, date, param, levelist=None, **kwargs) -> dict:
    """
    Download ``param`` for two time-steps (t-6h, t) and return a dict
    ``{variable_name: np.ndarray shape (2, N320_nodes)}``.
    """
    levelist = levelist or []
    raw: dict[str, list] = defaultdict(list)

    for t in [date - datetime.timedelta(hours=6), date]:
        dataset = ekd.from_source(
            "ecmwf-open-data",
            date=t,
            param=param,
            levelist=levelist,
            source=SOURCE,
            **kwargs,
        )
        for field in dataset:
            assert field.to_numpy().shape == (721, 1440), (
                f"Unexpected grid shape for {field.metadata('param')}: "
                f"{field.to_numpy().shape}"
            )
            # Shift lon from [0,360) to [-180,180) then regrid to N320 Gaussian
            values = np.roll(field.to_numpy(), -field.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})

            if levelist:
                name = f"{field.metadata('param')}_{field.metadata('levelist')}"
            else:
                name = field.metadata("param")
            raw[name].append(values)

    return {k: np.stack(v) for k, v in raw.items()}


def _build_fields(ekd, ekr, date: datetime.datetime) -> dict:
    """Download and transform all required fields for ``date``."""
    fields: dict = {}

    print("  ⬇  Surface fields …")
    fields.update(_fetch_fields(ekd, ekr, date, PARAM_SFC, levtype="sfc"))

    print("  ⬇  Wave fields …")
    fields.update(_fetch_fields(ekd, ekr, date, PARAM_WAVE, stream="wave"))

    print("  ⬇  Soil fields …")
    soil = _fetch_fields(ekd, ekr, date, PARAM_SOIL, levelist=SOIL_LEVELS)

    print("  ⬇  Pressure-level fields …")
    fields.update(_fetch_fields(ekd, ekr, date, PARAM_PL, levelist=LEVELS))

    # ── Transformations ───────────────────────────────────────────────────────

    # Wave direction: decompose scalar angle into sin/cos components
    mwd = fields.pop("mwd")
    mwd_rad = np.deg2rad(mwd)
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    # Rename soil fields to ECMWF short-names expected by AIFS
    _soil_rename = {
        "sot_1": "stl1",  "sot_2": "stl2",
        "vsw_1": "swvl1", "vsw_2": "swvl2",
    }
    for src, dst in _soil_rename.items():
        fields[dst] = soil[src]

    # Remove q levels that AIFS does not use
    fields.pop("q_10", None)
    fields.pop("q_50", None)

    # Apply land-sea mask to snow depth and soil moisture (ocean → NaN)
    try:
        lsm = ekd.from_source("file", "lsm.grib")[0].to_numpy(flatten=True)
        ocean_mask = np.equal(lsm, 0)
        for var in ("sd", "swvl1", "swvl2"):
            if var in fields:
                fields[var][:, ocean_mask] = np.nan
    except Exception:
        pass  # lsm.grib not found; skip masking

    # Convert geopotential height → geopotential  (Z = gh × g)
    G = 9.80665
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}", None)
        if gh is not None:
            fields[f"z_{level}"] = gh * G

    return fields


# ── Public API ────────────────────────────────────────────────────────────────

def load_ics(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> tuple[dict, datetime.datetime]:
    """
    Return ``(fields, date)`` for the latest available ECMWF Open Data run.

    Parameters
    ----------
    cache_dir:
        Directory where .npz caches are stored.
    force:
        Re-download even when a local cache exists.

    Returns
    -------
    fields:
        ``{variable_name: np.ndarray shape (2, N320_nodes)}``.
        The first axis indexes the two input time-steps: ``[t-6h, t]``.
    date:
        The forecast initialisation date/time (the *later* of the two
        time-steps).
    """
    import earthkit.data as ekd
    import earthkit.regrid as ekr
    from ecmwf.opendata import Client as OpendataClient

    ekd.config.set({"cache-policy": "user"})
    cache_dir = Path(cache_dir)

    date: datetime.datetime = OpendataClient(SOURCE).latest()
    print(f"📅  Latest ECMWF run: {date}")

    if not force:
        cached, path = _try_load(date, cache_dir)
        if cached is not None:
            sz_mb = path.stat().st_size / 1e6
            print(f"✅  Loaded from cache  ({sz_mb:.0f} MB)  →  {path}")
            return cached, date

    print("⬇️   Downloading initial conditions …")
    fields = _build_fields(ekd, ekr, date)

    path = _save(date, fields, cache_dir)
    sz_mb = path.stat().st_size / 1e6
    print(f"💾  Saved to {path}  ({sz_mb:.0f} MB)")

    return fields, date
