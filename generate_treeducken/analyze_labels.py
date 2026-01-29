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
import shutil

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

# # =========================
# # Specific Cospeciation value counts (open intervals)
# # =========================
# if "Cospeciations" in label_values:
#     arr = np.array(label_values["Cospeciations"], dtype=float)

#     c_low = int(np.sum((arr > 0.0) & (arr < 0.05)))
#     c_high = int(np.sum((arr > 0.95) & (arr < 1.0)))

#     print("\nCospeciation open-interval counts:")
#     print(f"  0 < Cospeciations < 0.05: {c_low}")
#     print(f"  0.95 < Cospeciations < 1.0: {c_high}")

#     # =========================
#     # Detailed tail showcase for Cospeciations (0, 0.1) and (0.9, 1.0)
#     # =========================
#     if "Cospeciations" in label_values:
#         tail_dir = os.path.join(hist_out_dir, "cospeciation_tails")
#         os.makedirs(tail_dir, exist_ok=True)

#         arr = np.array(label_values["Cospeciations"], dtype=float)

#         # Open intervals: exclude the endpoints as requested
#         low_tail = arr[(arr > 0.0) & (arr < 0.1)]
#         high_tail = arr[(arr > 0.9) & (arr < 1.0)]

#         def _save_tail_details(tail_arr, lo, hi, tag):
#             # 1) Save exact values (sorted) to CSV
#             sorted_vals = np.sort(tail_arr)
#             values_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_values.csv")
#             with open(values_csv, "w", newline="") as f:
#                 w = csv.writer(f)
#                 w.writerow(["rank", "value"])
#                 for i, v in enumerate(sorted_vals):
#                     w.writerow([i, float(v)])

#             # 2) Save rounded value counts (helps spot discretization effects)
#             rounded = np.round(sorted_vals, 6)
#             uniq, cnts = np.unique(rounded, return_counts=True)
#             counts_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_value_counts_round6.csv")
#             with open(counts_csv, "w", newline="") as f:
#                 w = csv.writer(f)
#                 w.writerow(["value_round6", "count"])
#                 for u, c in zip(uniq, cnts):
#                     w.writerow([float(u), int(c)])

#             # 3) Fine histogram in the tail range
#             # Choose a detailed bin width: 0.002 gives 50 bins across a 0.1 interval.
#             fine_bin = 0.002
#             fine_bins = np.arange(lo, hi + fine_bin, fine_bin)
#             counts, edges = np.histogram(tail_arr, bins=fine_bins)
#             centers = 0.5 * (edges[:-1] + edges[1:])

#             plt.figure(figsize=(10, 5))
#             plt.bar(centers, counts, width=fine_bin, align="center")
#             plt.xlabel(f"Cospeciations in ({lo}, {hi})")
#             plt.ylabel("Count")
#             plt.title(f"Cospeciations tail distribution ({lo} < x < {hi}) | bin={fine_bin}")
#             plt.tight_layout()

#             plot_path = os.path.join(tail_dir, f"Cospeciations_{tag}_hist_fine.png")
#             plt.savefig(plot_path, dpi=200)
#             plt.close()

#             # 4) Export fine-bin counts to CSV
#             fine_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_hist_fine_bins.csv")
#             with open(fine_csv, "w", newline="") as f:
#                 w = csv.writer(f)
#                 w.writerow(["bin_start", "bin_end", "count"])
#                 for i in range(len(counts)):
#                     w.writerow([float(edges[i]), float(edges[i + 1]), int(counts[i])])

#             # 5) Print a compact textual summary (quantiles) for quick inspection in logs
#             if sorted_vals.size > 0:
#                 qs = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
#                 qv = np.quantile(sorted_vals, qs)
#                 print(f"\nCospeciations {tag} tail summary ({lo} < x < {hi})")
#                 print(f"  n={sorted_vals.size} min={sorted_vals.min():.6f} max={sorted_vals.max():.6f} mean={sorted_vals.mean():.6f}")
#                 print("  quantiles:")
#                 for q, v in zip(qs, qv):
#                     print(f"    q={q:>4}: {float(v):.6f}")
#                 print(f"  Saved: {values_csv}")
#                 print(f"  Saved: {counts_csv}")
#                 print(f"  Saved: {plot_path}")
#                 print(f"  Saved: {fine_csv}")
#             else:
#                 print(f"\nCospeciations {tag} tail summary ({lo} < x < {hi})")
#                 print("  n=0 (no values in this open interval)")

#         _save_tail_details(low_tail, 0.0, 0.1, "low_0_0p1")
#         _save_tail_details(high_tail, 0.9, 1.0, "high_0p9_1")

# print("\nNaN values replaced with 0.0:")
# for label, cnt in nan_counts.items():
#     print(f"  {label}: {cnt}")


# Optional CSV output
out_csv = os.path.join("/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/", "test_label_stats.csv")
#out_csv = os.path.join("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "count", "mean", "std", "min", "max"])
    for label, s in stats.items():
        writer.writerow([label, s["count"], s["mean"], s["std"], s["min"], s["max"]])

print(f"\nSaved label statistics to: {out_csv}")
