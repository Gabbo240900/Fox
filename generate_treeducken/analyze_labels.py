#!/usr/bin/env python3
import os
import argparse
import csv
import math
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

# Ensure matplotlib has a writable config/cache dir on HPC nodes
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", f"matplotlib-{os.getuid()}")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from typing import Optional, Dict, List, Tuple


# -------------------------
# Robust key/value extraction from .tgl
# -------------------------
_KEYVAL_RE = re.compile(
    r"^(?P<key>[A-Za-z0-9_./\- ]+?)\s*[:=,\t ]\s*(?P<val>[-+]?((\d+\.?\d*)|(\.\d+))([eE][-+]?\d+)?|NaN|nan|None|none)$"
)

# Normalize possible variants found in files
_KEY_MAP = {
    "cospeciations": "Cospeciations",
    "cospeciation": "Cospeciations",
    "host_spread/switches": "Host_spread/Switches",
    "host_spread": "Host_spread/Switches",
    "switches": "Host_spread/Switches",
    "host_switches": "Host_spread/Switches",
    "hostswitches": "Host_spread/Switches",
}

EXPECTED = ["Cospeciations", "Host_spread/Switches"]


def _norm_key(k: str) -> str:
    k = k.strip().lower()
    # keep slash, remove spaces
    k = k.replace(" ", "")
    return k


def parse_events_from_tgl(path: str) -> Optional[Dict[str, float]]:
    """Parse a .tgl and return a dict with keys in EXPECTED (if found)."""
    events: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # First try strict key/value pattern
                m = _KEYVAL_RE.match(line)
                if m:
                    key_raw = m.group("key")
                    val_raw = m.group("val")
                    key = _KEY_MAP.get(_norm_key(key_raw))
                    if key is None:
                        continue
                    try:
                        if val_raw.lower() == "none":
                            v = float("nan")
                        else:
                            v = float(val_raw)
                    except Exception:
                        v = float("nan")
                    events[key] = v
                    continue

                # Fallback: a lot of generators write like: "Cospeciations 0.12" or "Host_Spread/Switches 0.03"
                parts = line.split()
                if len(parts) == 2:
                    k_try, v_try = parts
                    k_norm = _norm_key(k_try.replace("_", ""))
                    # accept underscore variants
                    if k_norm in ("cospeciations", "cospeciation"):
                        key = "Cospeciations"
                    elif k_norm in ("hostspread/switches", "hostspread", "switches", "hostswitches"):
                        key = "Host_spread/Switches"
                    else:
                        key = None
                    if key is None:
                        continue
                    try:
                        v_try_clean = v_try.replace(",", ".")
                        if v_try_clean.lower() == "none":
                            v = float("nan")
                        else:
                            v = float(v_try_clean)
                    except Exception:
                        v = float("nan")
                    events[key] = v

        return events
    except Exception:
        return None


def collect_values(tgl_files: List[str], workers: Optional[int]):
    if workers is None:
        workers = min(32, (os.cpu_count() or 1))

    values = {k: [] for k in EXPECTED}
    nan_counts = {k: 0 for k in EXPECTED}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(parse_events_from_tgl, p): p for p in tgl_files}
        for fut in as_completed(futs):
            p = futs[fut]
            ev = fut.result()
            if not ev:
                continue
            for k in EXPECTED:
                v = ev.get(k, float("nan"))
                # Treat missing/None/NaN as 0.0
                if v is None:
                    nan_counts[k] += 1
                    v = 0.0
                elif isinstance(v, float) and math.isnan(v):
                    nan_counts[k] += 1
                    v = 0.0
                values[k].append(float(v))

    return values, nan_counts


def save_hist(values: List[float], out_png: str, out_csv: str, title: str, xlabel: str, bins: int = 20):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        print(f"[WARN] No values for {xlabel}; skipping histogram.")
        return

    plt.figure(figsize=(8, 5))
    counts, edges = np.histogram(arr, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.bar(centers, counts, width=(edges[1] - edges[0]), align="center")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_start", "bin_end", "count"])
        for i in range(len(counts)):
            w.writerow([float(edges[i]), float(edges[i + 1]), int(counts[i])])


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of Cospeciations and Host_spread/Switches from .tgl datasets"
    )
    parser.add_argument(
        "--tgl_dir",
        type=str,
        required=True,
        help="Directory containing .tgl files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for histograms (default: <tgl_dir>/label_analysis/histograms)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of threads (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20)",
    )
    args = parser.parse_args()

    tgl_dir = args.tgl_dir
    if not os.path.isdir(tgl_dir):
        raise SystemExit(f"Input directory does not exist: {tgl_dir}")

    tgl_files = sorted(glob.glob(os.path.join(tgl_dir, "*.tgl")))
    if not tgl_files:
        raise SystemExit(f"No .tgl files found in: {tgl_dir}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(tgl_dir, "label_analysis", "histograms")
    os.makedirs(out_dir, exist_ok=True)

    values, nan_counts = collect_values(tgl_files, args.workers)

    # Print quick summary
    print(f"Loaded {len(tgl_files)} .tgl files")
    for k in EXPECTED:
        arr = np.array(values[k], dtype=float)
        if arr.size == 0:
            print(f"  {k}: n=0 (missing/NaN in {nan_counts[k]} files)")
        else:
            print(
                f"  {k}: n={arr.size} (missing/NaN in {nan_counts[k]} files) "
                f"mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}"
            )

    # Save histograms
    save_hist(
        values["Cospeciations"],
        os.path.join(out_dir, "Cospeciations_hist.png"),
        os.path.join(out_dir, "Cospeciations_hist_bins.csv"),
        title="Histogram of Cospeciations",
        xlabel="Cospeciations",
        bins=args.bins,
    )
    save_hist(
        values["Host_spread/Switches"],
        os.path.join(out_dir, "Host_spread_Switches_hist.png"),
        os.path.join(out_dir, "Host_spread_Switches_hist_bins.csv"),
        title="Histogram of Host_spread/Switches",
        xlabel="Host_spread/Switches",
        bins=args.bins,
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
