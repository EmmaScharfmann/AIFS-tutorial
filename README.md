---
title: "Running ECMWF AIFS on Any Machine — No Ampere GPU Required"
thumbnail: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aifs-tutorial/thumbnail.png
authors:
  - user: your-hf-username
tags:
  - weather
  - climate
  - aifs
  - anemoi
  - ecmwf
  - forecasting
---

# Running ECMWF AIFS on Any Machine — No Ampere GPU Required

ECMWF's [AIFS](https://www.ecmwf.int/en/about/media-centre/aifs-machine-learning-weather-model)
(Artificial Intelligence Forecast System) is one of the most accurate
operational weather models ever built.  It consistently outperforms
traditional numerical weather prediction at medium range and runs in
seconds rather than hours.

The problem?  The official implementation depends on
[`flash-attn`](https://github.com/Dao-AILab/flash-attention), a library
that only compiles on **Ampere-class NVIDIA GPUs** (A100, H100, RTX 30xx+).
For most researchers and practitioners — especially those working on
older clusters, Apple Silicon, or CPU-only machines — this is a hard wall.

This tutorial shows you how to get around it with a **pure-PyTorch SDPA
shim** that makes AIFS run on any hardware.

---

## Why AIFS Needs flash-attn (And Why It Doesn't Have To)

AIFS uses a **sliding-window graph-transformer** architecture.  Each
attention layer calls `flash_attn_func` from the `flash-attn` package,
which fuses the softmax and matrix multiplications into a single CUDA
kernel — fast, memory-efficient, and Ampere-only.

PyTorch 2.1+ ships its own fused attention via
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which:

- Dispatches to a flash-attn-style kernel on capable hardware
- Falls back gracefully to memory-efficient attention or naive attention
- Works on CUDA, Apple MPS, and CPU

Our shim intercepts Anemoi's `flash_attn` import and routes it to SDPA
— no recompilation, no CUDA toolkit required.

---

## Setup

```bash
git clone https://huggingface.co/datasets/YOUR_USERNAME/aifs-tutorial
cd aifs-tutorial
pip install -r requirements.txt
```

> You do **not** need to `pip install flash-attn`.

**requirements.txt** (key packages):
```
torch>=2.1.0
anemoi-inference>=0.4
earthkit-data>=0.10
earthkit-regrid>=0.2
ecmwf-opendata>=0.3
cartopy>=0.22
```

---

## How the Shim Works

The shim lives in `aifs/compat.py`.  It creates a fake `flash_attn`
module tree and registers it in `sys.modules` *before* any Anemoi import
can trigger the real package lookup:

```python
# aifs/compat.py  (simplified)
import sys, types
import torch.nn.functional as F

def _sdpa_compat(q, k, v, window_size=(-1,-1), dropout_p=0.0):
    # flash-attn layout:  (B, S, H, D)
    # SDPA layout:        (B, H, S, D)
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    ws = window_size[0]

    if q.device.type == "cuda":
        out = F.scaled_dot_product_attention(q, k, v)

    elif ws > 0:
        # MPS: chunked sliding-window to avoid OOM
        B, H, S, D = q.shape
        out = torch.zeros_like(q)
        for i in range(0, S, ws):
            out[:, :, i:i+ws] = F.scaled_dot_product_attention(
                q[:, :, i:i+ws],
                k[:, :, max(0,i-ws):min(S,i+2*ws)],
                v[:, :, max(0,i-ws):min(S,i+2*ws)],
            )
    else:
        out = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())

    return out.permute(0, 2, 1, 3)

# Register stub before any anemoi import
fake_mod = types.ModuleType("flash_attn.flash_attn_interface")
fake_mod.flash_attn_func = _sdpa_compat
sys.modules["flash_attn.flash_attn_interface"] = fake_mod
```

This is the entire trick.  Everything else is just plumbing.

---

## Step-by-Step Tutorial

### 1. Check Your Device

```python
from aifs import get_device, device_label
print(device_label())
# → "CUDA — NVIDIA A100-SXM4-40GB"
# → "Apple MPS (Metal)"
# → "CPU (no GPU detected — inference will be slow)"
```

### 2. Download Initial Conditions

AIFS needs two consecutive 6-hour analyses as input: **t-6h** and **t**.
We pull them from [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
— free, no account required.

```python
from aifs import load_ics

fields, date = load_ics(cache_dir="../ic_cache")
# First run: ~3–5 min download, saved to ic_cache/
# Later runs: <1 s from local .npz cache

print(date)  # 2025-01-15 00:00:00
print(fields["2t"].shape)  # (2, 542080)  — two time-steps on the N320 grid
```

The function returns a plain dict of NumPy arrays.  Each array has shape
`(2, N_nodes)` where the first axis indexes the two input time-steps and
`N_nodes ≈ 542,080` is the size of the N320 reduced-Gaussian grid.

**Fields downloaded**: surface (13 variables), soil (4), ocean waves (11),
pressure levels (5 variables × 14 levels = 70), plus derived quantities
(geopotential, wave direction components).

### 3. Run a Forecast

```python
from aifs import run_forecast

states = run_forecast(
    fields,
    date,
    lead_time=24,   # hours; must be multiple of 6
    num_chunks=16,  # tune for your hardware (see table below)
)
```

**Memory guide:**

| RAM / VRAM | `num_chunks` |
|---|---|
| ≥ 40 GB | 4 |
| 16–24 GB | 16 |
| 8–12 GB | 32 |
| < 8 GB / CPU | 64 |

Each call to `run_forecast` returns a list of state dicts — one per
6-hour output step:

```python
for state in states:
    print(state["date"], state["fields"]["2t"].mean() - 273.15, "°C")
# 2025-01-15 06:00:00   14.32 °C
# 2025-01-15 12:00:00   14.38 °C
# 2025-01-15 18:00:00   14.41 °C
# 2025-01-16 00:00:00   14.29 °C
```

### 4. Plot Forecast Fields

```python
from aifs import plot_field, plot_field_sequence

# Single map
fig = plot_field(states[0], "2t")
fig.savefig("t2m_T+6h.png", dpi=150)

# Four-panel sequence
fig = plot_field_sequence(states, "msl", max_steps=4)
fig.savefig("msl_sequence.png", dpi=150)
```

Available fields for plotting:

```python
from aifs import PLOTTABLE
print(PLOTTABLE)
# ['2t', 'msl', 'sp', 'tcw', '10u', '10v', 'swh', 'mwp',
#  't_850', 't_500', 'u_850', 'v_850', 'z_500', 'q_700']
```

### 5. CLI Usage

```bash
# 24-hour forecast, save plots to ./outputs/
python run_forecast.py

# 48-hour forecast
python run_forecast.py --lead-time 48 --fields 2t msl z_500

# List IC cache
python run_forecast.py --list-cache

# Force re-download
python run_forecast.py --force-download
```

### 6. Streaming for Long Runs

```python
from aifs import run_forecast_streaming

for state in run_forecast_streaming(fields, date, lead_time=120):
    t = state["fields"]["2t"].mean() - 273.15
    print(f"{state['date']}  global mean T2m = {t:.2f} °C")
```

---

## Performance Expectations

| Hardware | Time per 6h step | Notes |
|---|---|---|
| A100 / H100 | ~5–10 s | Near-operational speed |
| RTX 3090 / 4090 | ~15–25 s | |
| RTX 2080 Ti | ~30–50 s | Older Turing GPU, no flash-attn |
| Apple M2 Pro | ~3–8 min | MPS chunked attention |
| Modern CPU (16-core) | ~15–40 min | Practical for testing / CI |

---

## Repository Structure

```
aifs-tutorial/
├── aifs/
│   ├── __init__.py          # Clean public API
│   ├── compat.py            # flash-attn → SDPA shim  ← the key piece
│   ├── device.py            # Device detection helpers
│   ├── initial_conditions.py # ECMWF Open Data download + cache
│   ├── forecast.py          # Thin anemoi-inference wrapper
│   └── plot.py              # Cartopy-based visualisation
├── notebooks/
│   └── tutorial.py          # This tutorial as a runnable notebook
├── run_forecast.py           # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Frequently Asked Questions

**Q: Do I need an ECMWF account?**
No.  We use [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data),
which is freely available without registration under the CC-BY 4.0 license.

**Q: Can I use AIFS Ensemble instead of AIFS Single?**
Yes.  In `aifs/forecast.py`, add an entry to `CHECKPOINTS`:
```python
"aifs-ens-1.0": {"huggingface": "ecmwf/aifs-ens-1.0"},
```
Then pass `checkpoint="aifs-ens-1.0"` to `run_forecast()`.

**Q: The model checkpoint is huge — where is it downloaded?**
Hugging Face caches it under `~/.cache/huggingface/hub/`.
The AIFS Single v2 checkpoint is ~4 GB.

**Q: How do I export to NetCDF for use with xarray?**
Regrid from N320 back to a regular lat/lon grid with `earthkit-regrid`,
then write with `xarray` + `netCDF4`:
```python
import earthkit.regrid as ekr
import xarray as xr
import numpy as np

# Regrid one field from N320 to 0.25° regular grid
data_ll = ekr.interpolate(
    states[0]["fields"]["2t"],
    {"grid": "N320"},
    {"grid": (0.25, 0.25)},
)
lats = np.arange(90, -90.25, -0.25)
lons = np.arange(0, 360, 0.25)
da = xr.DataArray(data_ll, dims=["lat", "lon"],
                  coords={"lat": lats, "lon": lons})
da.to_netcdf("t2m_T+6h.nc")
```

**Q: My MPS run crashes with an out-of-memory error.**
Increase `num_chunks` (e.g. 64 or 128) and make sure no other GPU
workloads are running.

---

## Citation

If you use AIFS in your research, please cite the ECMWF technical note:

```bibtex
@techreport{lang2024aifs,
  title   = {AIFS -- ECMWF's data-driven forecasting system},
  author  = {Lang, Simon and others},
  year    = {2024},
  institution = {ECMWF},
  url     = {https://arxiv.org/abs/2406.01465}
}
```

---

*This tutorial is community-contributed and is not officially affiliated
with ECMWF.  The AIFS model weights are distributed by ECMWF under their
own license.*
