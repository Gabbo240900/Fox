#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
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
args = parser.parse_args()


class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir):
        self.pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
        self.preencoded_dir = preencoded_dir

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)

        return sample.get("event_frequencies", {})


preencoded_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/preencoded_pt/"
#preencoded_dir = '/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/test/'
dataset = LazyCophyloformerDataset(preencoded_dir)

EXPECTED_LABELS = [
    "Cospeciations",
    "Host_spread/Switches",
    "Sim_time",
]

def read_event_frequencies(pt_path):
    try:
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        return sample.get("event_frequencies", {})
    except Exception as e:
        return None

label_values = defaultdict(list)
nan_counts = defaultdict(int)

# =========================
# Extract extreme cospeciation datasets (exactly 0 or exactly 1)
# =========================
EXTREME_COSP_DIR = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data"
# EXTREME_COSP_DIR = "/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/extreme_cosp_data"

extreme_cosp_0 = []
extreme_cosp_1 = []

num_samples = len(dataset.pt_files)

if args.workers is not None:
    max_workers = args.workers
else:
    max_workers = min(32, os.cpu_count() or 1)

print(f"Using {max_workers} threads for label extraction")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_path = {
        executor.submit(read_event_frequencies, pt_path): pt_path
        for pt_path in dataset.pt_files
    }

    for fut in as_completed(future_to_path):
        pt_path = future_to_path[fut]
        events = fut.result()
        if not events:
            continue

        # Track extreme cospeciation datasets (exact equality requested)
        cosp = events.get("Cospeciations", 0.0)
        if isinstance(cosp, float) and math.isnan(cosp):
            cosp = 0.0
        if cosp == 0.0:
            extreme_cosp_0.append(pt_path)
        elif cosp == 1.0:
            extreme_cosp_1.append(pt_path)

        # Fill missing labels with 0.0
        for label in EXPECTED_LABELS:
            if label in events:
                v = events[label]
                if isinstance(v, float) and math.isnan(v):
                    nan_counts[label] += 1
                    v = 0.0
            else:
                # Label missing entirely → interpret as 0.0
                v = 0.0
                nan_counts[label] += 1

            label_values[label].append(v)

print("\nSanity check (counts should equal number of samples):")
for label in EXPECTED_LABELS:
    print(f"  {label}: {len(label_values[label])} / {num_samples}")

def count_values_in_range(values, target, tol=1e-6):
    """
    Count how many values fall within [target - tol, target + tol].
    """
    arr = np.array(values, dtype=float)
    return int(np.sum((arr >= target - tol) & (arr <= target + tol)))

stats = {}
for label, values in label_values.items():
    arr = np.array(values, dtype=float)
    stats[label] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

# =========================
# Histogram plots (bins of 0.05)
# =========================
hist_out_dir = "/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/test_histograms"
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

# =========================
# Specific Cospeciation value counts (open intervals)
# =========================
if "Cospeciations" in label_values:
    arr = np.array(label_values["Cospeciations"], dtype=float)

    c_low = int(np.sum((arr > 0.0) & (arr < 0.05)))
    c_high = int(np.sum((arr > 0.95) & (arr < 1.0)))

    print("\nCospeciation open-interval counts:")
    print(f"  0 < Cospeciations < 0.05: {c_low}")
    print(f"  0.95 < Cospeciations < 1.0: {c_high}")

    # =========================
    # Detailed tail showcase for Cospeciations (0, 0.1) and (0.9, 1.0)
    # =========================
    if "Cospeciations" in label_values:
        tail_dir = os.path.join(hist_out_dir, "cospeciation_tails")
        os.makedirs(tail_dir, exist_ok=True)

        arr = np.array(label_values["Cospeciations"], dtype=float)

        # Open intervals: exclude the endpoints as requested
        low_tail = arr[(arr > 0.0) & (arr < 0.1)]
        high_tail = arr[(arr > 0.9) & (arr < 1.0)]

        def _save_tail_details(tail_arr, lo, hi, tag):
            # 1) Save exact values (sorted) to CSV
            sorted_vals = np.sort(tail_arr)
            values_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_values.csv")
            with open(values_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "value"])
                for i, v in enumerate(sorted_vals):
                    w.writerow([i, float(v)])

            # 2) Save rounded value counts (helps spot discretization effects)
            rounded = np.round(sorted_vals, 6)
            uniq, cnts = np.unique(rounded, return_counts=True)
            counts_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_value_counts_round6.csv")
            with open(counts_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["value_round6", "count"])
                for u, c in zip(uniq, cnts):
                    w.writerow([float(u), int(c)])

            # 3) Fine histogram in the tail range
            # Choose a detailed bin width: 0.002 gives 50 bins across a 0.1 interval.
            fine_bin = 0.002
            fine_bins = np.arange(lo, hi + fine_bin, fine_bin)
            counts, edges = np.histogram(tail_arr, bins=fine_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])

            plt.figure(figsize=(10, 5))
            plt.bar(centers, counts, width=fine_bin, align="center")
            plt.xlabel(f"Cospeciations in ({lo}, {hi})")
            plt.ylabel("Count")
            plt.title(f"Cospeciations tail distribution ({lo} < x < {hi}) | bin={fine_bin}")
            plt.tight_layout()

            plot_path = os.path.join(tail_dir, f"Cospeciations_{tag}_hist_fine.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()

            # 4) Export fine-bin counts to CSV
            fine_csv = os.path.join(tail_dir, f"Cospeciations_{tag}_hist_fine_bins.csv")
            with open(fine_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_start", "bin_end", "count"])
                for i in range(len(counts)):
                    w.writerow([float(edges[i]), float(edges[i + 1]), int(counts[i])])

            # 5) Print a compact textual summary (quantiles) for quick inspection in logs
            if sorted_vals.size > 0:
                qs = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
                qv = np.quantile(sorted_vals, qs)
                print(f"\nCospeciations {tag} tail summary ({lo} < x < {hi})")
                print(f"  n={sorted_vals.size} min={sorted_vals.min():.6f} max={sorted_vals.max():.6f} mean={sorted_vals.mean():.6f}")
                print("  quantiles:")
                for q, v in zip(qs, qv):
                    print(f"    q={q:>4}: {float(v):.6f}")
                print(f"  Saved: {values_csv}")
                print(f"  Saved: {counts_csv}")
                print(f"  Saved: {plot_path}")
                print(f"  Saved: {fine_csv}")
            else:
                print(f"\nCospeciations {tag} tail summary ({lo} < x < {hi})")
                print("  n=0 (no values in this open interval)")

        _save_tail_details(low_tail, 0.0, 0.1, "low_0_0p1")
        _save_tail_details(high_tail, 0.9, 1.0, "high_0p9_1")

print("\nNaN values replaced with 0.0:")
for label, cnt in nan_counts.items():
    print(f"  {label}: {cnt}")

# =========================
# Copy extreme cospeciation datasets into EXTREME_COSP_DIR
# =========================
os.makedirs(EXTREME_COSP_DIR, exist_ok=True)

c0_dir = os.path.join(EXTREME_COSP_DIR, "cosp_0")
c1_dir = os.path.join(EXTREME_COSP_DIR, "cosp_1")
os.makedirs(c0_dir, exist_ok=True)
os.makedirs(c1_dir, exist_ok=True)

print("\nExtreme cospeciation extraction:")
print(f"  Cospeciations == 0.0: {len(extreme_cosp_0)}")
print(f"  Cospeciations == 1.0: {len(extreme_cosp_1)}")

manifest_csv = os.path.join(EXTREME_COSP_DIR, "manifest.csv")
with open(manifest_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source_pt", "target_pt", "cospeciations_value"])

    for src in extreme_cosp_0:
        dst = os.path.join(c0_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        w.writerow([src, dst, 0.0])

    for src in extreme_cosp_1:
        dst = os.path.join(c1_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        w.writerow([src, dst, 1.0])

print(f"  Copied files into: {EXTREME_COSP_DIR}")
print(f"  Wrote manifest: {manifest_csv}")

# Optional CSV output
out_csv = os.path.join("/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/", "test_label_stats.csv")
#out_csv = os.path.join("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")



with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "count", "mean", "std", "min", "max"])
    for label, s in stats.items():
        writer.writerow([label, s["count"], s["mean"], s["std"], s["min"], s["max"]])

print(f"\nSaved label statistics to: {out_csv}")
