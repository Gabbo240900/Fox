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

def read_event_frequencies(pt_path):
    try:
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        return sample.get("event_frequencies", {})
    except Exception as e:
        return None

label_values = defaultdict(list)

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
        for k, v in events.items():
            if isinstance(v, (int, float)):
                label_values[k].append(v)

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

# Print to stdout
for label, s in stats.items():
    print(
        f"{label}: count={s['count']} "
        f"mean={s['mean']:.6f} std={s['std']:.6f} "
        f"min={s['min']:.6f} max={s['max']:.6f}"
    )

# Optional CSV output
out_csv = os.path.join("/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")
#out_csv = os.path.join("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/label_analysis/", "label_stats.csv")



with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "count", "mean", "std", "min", "max"])
    for label, s in stats.items():
        writer.writerow([label, s["count"], s["mean"], s["std"], s["min"], s["max"]])

print(f"\nSaved label statistics to: {out_csv}")
