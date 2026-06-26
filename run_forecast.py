#!/usr/bin/env python3
"""
run_forecast.py — CLI wrapper for AIFS forecasting
===================================================

Usage examples
--------------
# Basic 24-hour forecast, save plots to ./outputs/
python run_forecast.py

# 48-hour forecast with custom output directory
python run_forecast.py --lead-time 48 --output-dir my_results

# Force re-download of initial conditions
python run_forecast.py --force-download

# List what's already in the IC cache
python run_forecast.py --list-cache

Full option reference
---------------------
    --lead-time      Forecast horizon in hours (default: 24, must be multiple of 6)
    --num-chunks     Attention chunks; increase to save memory (default: 16)
    --fields         Space-separated list of fields to plot (default: 2t msl tcw swh)
    --output-dir     Where to write PNG files (default: ./outputs)
    --cache-dir      Where to store / read IC caches (default: ./ic_cache)
    --force-download Re-download ICs even when a cache exists
    --list-cache     Print cached IC dates and exit
    --no-plots       Skip plotting (useful for debugging / memory profiling)
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run an AIFS weather forecast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lead-time",      type=int,  default=24,
                   help="Forecast horizon in hours (multiple of 6)")
    p.add_argument("--num-chunks",     type=int,  default=16,
                   help="Attention chunks (increase to reduce VRAM usage)")
    p.add_argument("--fields",         nargs="+", default=["2t", "msl", "tcw", "swh"],
                   help="Fields to plot")
    p.add_argument("--output-dir",     type=Path, default=Path("outputs"),
                   help="Directory for output PNG files")
    p.add_argument("--cache-dir", type=Path, default=Path("../ic_cache"),
                   help="Directory for IC .npz cache files")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download ICs even when a local cache exists")
    p.add_argument("--list-cache",     action="store_true",
                   help="Print cached IC dates and exit")
    p.add_argument("--no-plots",       action="store_true",
                   help="Skip generating plots")
    return p.parse_args()


def main():
    args = parse_args()

    # Lazy import so --help is fast even without heavy deps installed
    import warnings
    warnings.filterwarnings("ignore")

    from aifs.initial_conditions import load_ics, list_cached
    from aifs.device import  device_label
    from aifs.forecast import run_forecast
    from aifs.plot import  plot_field_sequence

    # ── List cache and exit ────────────────────────────────────────────────────
    if args.list_cache:
        cached = list_cached(args.cache_dir)
        if not cached:
            print("No cached ICs found in", args.cache_dir)
        else:
            print(f"Cached ICs in {args.cache_dir}:")
            for path in cached:
                sz = path.stat().st_size / 1e6
                print(f"  • {path.stem.replace('ic_', '')}  ({sz:.0f} MB)")
        sys.exit(0)

    # ── Main forecast pipeline ─────────────────────────────────────────────────
    print("=" * 60)
    print("  AIFS Forecast Runner")
    print("=" * 60)
    print(f"  Device     : {device_label()}")
    print(f"  Lead time  : {args.lead_time} h")
    print(f"  Num chunks : {args.num_chunks}")
    print(f"  Fields     : {args.fields}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    # 1. Initial conditions
    fields, date = load_ics(
        cache_dir=args.cache_dir,
        force=args.force_download,
    )

    # 2. Run forecast
    states = run_forecast(
        fields,
        date,
        lead_time=args.lead_time,
        num_chunks=args.num_chunks,
    )

    # 3. Save plots
    if not args.no_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for variable in args.fields:
            try:
                fig = plot_field_sequence(states, variable, max_steps=4)
                out = args.output_dir / f"{variable}_{date.strftime('%Y%m%dT%H%M%S')}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"  📊  Saved  →  {out}")
            except Exception as exc:
                print(f"  ⚠️   Could not plot '{variable}': {exc}")

    print(f"\n✅  Forecast complete.  {len(states)} steps produced.")


if __name__ == "__main__":
    main()
