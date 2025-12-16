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

num_samples = len(dataset.pt_files)

if args.workers is not None:
    max_workers = args.workers
else:
    max_workers = min(32, os.cpu_count() or 1)

print(f"Using {max_workers} threads for label extraction")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(read_event_frequencies, pt_path)
        for pt_path in dataset.pt_files
    ]

    for fut in as_completed(futures):
        events = fut.result()
        if not events:
            continue
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

print("\nNaN values replaced with 0.0:")
for label, cnt in nan_counts.items():
    print(f"  {label}: {cnt}")

# Optional CSV output
out_csv = os.path.join("/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")
#out_csv = os.path.join("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")



with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "count", "mean", "std", "min", "max"])
    for label, s in stats.items():
        writer.writerow([label, s["count"], s["mean"], s["std"], s["min"], s["max"]])

print(f"\nSaved label statistics to: {out_csv}")
