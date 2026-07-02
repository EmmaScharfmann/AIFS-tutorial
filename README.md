# Running ECMWF AIFS using CUDA or Apple MPS or CPU ![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

ECMWF has released its AIFS weights on Hugging Face, including the current [aifs-single-2.0](https://huggingface.co/ecmwf/aifs-single-2.0) checkpoint. 
AIFS is a data-driven medium-range forecasting model trained on ECMWF's ERA5 reanalysis and operational NWP analyses, and is described in [Lang et al. (2024)](https://arxiv.org/abs/2406.01465).

To generate a forecast from the published weights, ECMWF's documentation points to ```anemoi-inference```.
However, ```anemoi-inference``` default installation depends on [`flash-attn`](https://github.com/Dao-AILab/flash-attention), 
which only compiles on **Ampere-class NVIDIA GPUs** (A100, H100, RTX 30xx+).
For anyone without that specific hardware, installing ```flash-attn``` and dealing with ```anemoi``` dependencies are the main barriers to running inference.

Therefore, we developed this tutorial and a lightweight wrapper to simplify running [aifs-single-2.0](https://huggingface.co/ecmwf/aifs-single-2.0) with ```anemoi-inference``` on any hardware.

---

## Why does ```anemoi-inference``` rely on  ```flash-attn```

AIFS uses a **sliding-window graph-transformer** architecture.  Each
attention layer calls `flash_attn_func` from the `flash-attn` package,
which computes the softmax and matrix multiplications into a single CUDA
kernel — fast, memory-efficient, and **Ampere-only**.

PyTorch 2.1+ ships its own fused attention via
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which:

- Dispatches to a flash-attn-style kernel on capable hardware
- Falls back gracefully to memory-efficient attention or naive attention
- Works on CUDA, Apple MPS, and CPU

Our wrapper intercepts Anemoi's `flash_attn` import and routes it to SDPA.

---

## Setup

**Requirements:** Python ≥3.11, <3.13

```bash
git clone https://github.com/EmmaScharfmann/AIFS-tutorial
cd aifs-tutorial
pip install -r requirements.txt
```

> You do **not** need to `pip install flash-attn`.

**requirements.txt** (key packages):
```
torch==2.11.0
anemoi-models==0.9.3
anemoi-transform==0.4.2
anemoi-datasets==0.5.26
anemoi-graphs==0.6.4
anemoi-inference
```

---

## How it works

The wrapper lives in `aifs/compat.py`.  It creates a fake `flash_attn`
module tree and registers it in `sys.modules` *before* any Anemoi import
can trigger the real package lookup:

```python
def _patch():
    """Install the flash_attn stub into ``sys.modules``."""
    if "flash_attn" in sys.modules:
        return

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__version__ = "2.6.0"  
    flash_attn.flash_attn_func = _sdpa_compat  

    # flash_attn.layers.rotary  (imported but only used on specific GPU paths)
    layers_mod = types.ModuleType("flash_attn.layers")
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")

    def _rotary_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn.layers.rotary.RotaryEmbedding is not available in the SDPA "
            "compatibility shim; this code path requires real flash-attn on CUDA."
        )

    rotary_mod.RotaryEmbedding = _rotary_not_implemented
    layers_mod.rotary = rotary_mod
    flash_attn.layers = layers_mod

    # flash_attn.flash_attn_interface  (the one Anemoi actually calls)
    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.flash_attn_func = _sdpa_compat
    flash_attn.flash_attn_interface = interface_mod

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod


_patch()
```

This code allows to run inference of AIFS on any GPU or CPU. 

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

Multiple datasets are downloaded at this step. Once downloaded for the first time, each dataset is stored in cache. 
The first download takes a few minutes. 
Then, loading the data from the cache only takes a few seconds. 

During the initial download, ratelimits and errors sometimes occur. If you're getting an error during the downloading of the data, just wait a few seconds and try again. 
```python
from aifs import load_ics

fields, date = load_ics(cache_dir="../ic_cache")
# First run: ~3–5 min download, saved to ic_cache/
# Later runs: <1 s from local .npz cache

print(date)  
print(fields["2t"].shape)
```

**Fields downloaded**: surface, soil, ocean waves,
pressure levels, plus derived quantities (geopotential, wave direction components).

### 3. Run a Forecast

```python
from aifs import run_forecast

states = run_forecast(
    fields,
    date,
    lead_time=24,  
    num_chunks=16, 
)
```

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
├── forecasts/               # Where your forecasts will be saved
├── run_forecast.py           # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## FAQ

**Q: Where does the initial data come from?**
We use [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data),
which is freely available without registration under the CC-BY 4.0 license.

**Q: Can I use AIFS Ensemble instead of AIFS Single?**
Yes, the compat.py patch method can be applied to AIFS ensemble.  In `aifs/forecast.py`, add an entry to `CHECKPOINTS`:
```python
"aifs-ens-2.0": {"huggingface": "ecmwf/aifs-ens-2.0"},
```
Then pass `checkpoint="aifs-ens-2.0"` to `run_forecast()`.
However, the checkpoints were not trained with the same version of the different anemoi-packages. The requirements must be adapted accordingly.  

**Q: What is the size of the AIFs model and where is it stored?**
Hugging Face caches it under `~/.cache/huggingface/hub/`.
The AIFS Single v2 checkpoint is ~4 GB.

**Q: My MPS run crashes with an out-of-memory error.**
Increase `num_chunks` (e.g. 64 or 128) to reduce the memory usage and make sure no other GPU
workloads are running.

---


*This tutorial is community-contributed and is not officially affiliated
with ECMWF.  The AIFS model weights are distributed by ECMWF under their
own license.*
