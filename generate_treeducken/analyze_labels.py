#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import re
from collections import defaultdict
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import matplotlib.pyplot as plt


# Investigate cospeciation 1 scenarios - also cospeciation  0

parser = argparse.ArgumentParser(description="Compute mean/std of labels from pre-encoded .pt datasets")
parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Number of worker threads to use (default: min(32, CPU count))",
)
parser.add_argument(
    "--tgl_dir",
    type=str,
    default="/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/test/",
    help="Directory containing .tgl files",
)
args = parser.parse_args()


class LazyTGLDataset:
    def __init__(self, tgl_dir):
        self.tgl_files = sorted(glob.glob(os.path.join(tgl_dir, "*.tgl")))
        self.tgl_dir = tgl_dir

    def __len__(self):
        return len(self.tgl_files)

    def __getitem__(self, idx):
        # returns parsed event frequencies dict
        return parse_event_frequencies_from_tgl(self.tgl_files[idx])


# -------------------------
# TGL parsing utilities
# -------------------------
_KEYVAL_RE = re.compile(
    r"^(?P<key>[A-Za-z0-9_./\- ]+?)\s*[:=,\t]\s*(?P<val>[-+]?((\d+\.?\d*)|(\.\d+))([eE][-+]?\d+)?|NaN|nan)$"
)

# Common places where generators store stats/params. We stay permissive and just
# look for key/value lines anywhere in the file.
_EXPECTED_KEYS_NORMALIZED = {
    "cospeciations": "Cospeciations",
    "host_spread/switches": "Host_spread/Switches",
    "host_spread": "Host_spread/Switches",
    "switches": "Host_spread/Switches",
    "sim_time": "Sim_time",
    "simtime": "Sim_time",
}

def _to_float_or_nan(s: str):
    try:
        return float(s)
    except Exception:
        return float("nan")

def parse_event_frequencies_from_tgl(tgl_path: str):
    """Parse a .tgl file and return a dict of event frequency labels.

    This is intentionally robust: it scans the file for key/value pairs and maps
    common synonyms to the expected label names.
    """
    events = {}
    try:
        with open(tgl_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                m = _KEYVAL_RE.match(line)
                if not m:
                    continue

                key_raw = m.group("key").strip()
                val_raw = m.group("val").strip()

                key_norm = key_raw.lower().replace(" ", "").replace("-", "").replace("_", "_")
                # keep slashes for host_spread/switches
                key_norm = key_norm.replace("__", "_")

                # Also accept keys with spaces but normalize similarly
                key_norm2 = key_raw.lower().strip()
                key_norm2 = key_norm2.replace(" ", "").replace("-", "").replace("_", "_")

                mapped = (
                    _EXPECTED_KEYS_NORMALIZED.get(key_norm2)
                    or _EXPECTED_KEYS_NORMALIZED.get(key_raw.lower().strip())
                    or _EXPECTED_KEYS_NORMALIZED.get(key_norm)
                )

                if mapped is None:
                    continue

                v = _to_float_or_nan(val_raw)
                events[mapped] = v

        return events
    except Exception:
        return None


def read_event_frequencies(tgl_path):
    return parse_event_frequencies_from_tgl(tgl_path)


tgl_dir = args.tgl_dir
if not os.path.isdir(tgl_dir):
    raise SystemExit(f"Input directory does not exist: {tgl_dir}")

dataset = LazyTGLDataset(tgl_dir)

EXPECTED_LABELS = [
    "Cospeciations",
    "Host_spread/Switches",
    "Sim_time",
]

label_values = defaultdict(list)
nan_counts = defaultdict(int)

# Load labels from .tgl files
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
        executor.submit(parse_event_frequencies_from_tgl, tgl_path): tgl_path
        for tgl_path in dataset.tgl_files
    }
    for future in as_completed(futures):
        tgl_path = futures[future]
        events = future.result()
        if events is None:
            continue

        # Track NaN values and replace with 0.0
        for label in EXPECTED_LABELS:
            v = events.get(label, float("nan"))
            if isinstance(v, float) and math.isnan(v):
                nan_counts[label] += 1
                v = 0.0
            label_values[label].append(v)

# Compute statistics for each label
stats = {}
for label, values in label_values.items():
    arr = np.array(values, dtype=float)
    stats[label] = {
        "count": len(arr),
        "mean": np.mean(arr) if len(arr) > 0 else float("nan"),
        "std": np.std(arr) if len(arr) > 0 else float("nan"),
        "min": np.min(arr) if len(arr) > 0 else float("nan"),
        "max": np.max(arr) if len(arr) > 0 else float("nan"),
    }

# =========================
# Histogram plots (bins of 0.05)
# =========================
hist_out_dir = "/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/histograms"
#hist_out_dir = "/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/histograms"
os.makedirs(hist_out_dir, exist_ok=True)

bin_width = 0.05
bins = np.arange(0.0, 1.0 + bin_width, bin_width)

for label, values in label_values.items():
    arr = np.array(values, dtype=float)

    plt.figure(figsize=(8, 5))
    counts, edges = np.histogram(arr, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.bar(centers, counts, width=bin_width, align="center")
    for c, x in zip(counts, centers):
        if c > 0:
            plt.text(
                x,
                c,
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    plt.xlabel(label)
    plt.ylabel("Count")
    plt.title(f"Histogram of {label} (bin width = {bin_width})")
    plt.tight_layout()

    out_path = os.path.join(hist_out_dir, f"{label.replace('/', '_')}_hist.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved histogram for {label} to: {out_path}")

    csv_path = os.path.join(hist_out_dir, f"{label.replace('/', '_')}_hist_bins.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start", "bin_end", "count"])
        for i in range(len(counts)):
            writer.writerow([edges[i], edges[i+1], int(counts[i])])
    print(f"Saved histogram bin counts for {label} to: {csv_path}")

# Print to stdout
for label, s in stats.items():
    print(
        f"{label}: count={s['count']} "
        f"mean={s['mean']:.6f} std={s['std']:.6f} "
        f"min={s['min']:.6f} max={s['max']:.6f}"
    )

# Optional CSV output
out_csv = os.path.join("/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/", "test_label_stats.csv")
#out_csv = os.path.join("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "count", "mean", "std", "min", "max"])
    for label, s in stats.items():
        writer.writerow([label, s["count"], s["mean"], s["std"], s["min"], s["max"]])

print(f"\nSaved label statistics to: {out_csv}")
