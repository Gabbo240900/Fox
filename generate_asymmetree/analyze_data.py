#!/usr/bin/env python3
"""
analyze_data.py — unified dataset analysis tool for Co-Phyloformer datasets.

Outputs:
  • Tree structure: host / parasite leaf counts and alignment lengths
  • Event frequencies: Speciation, HGT, Loss, Duplication

Supported input formats:
  • .tgl / .nex / .nexus  — raw NEXUS-format files (asymmetree / treeducken)
  • .pt                   — pre-encoded PyTorch tensors produced by data.py

Usage:
  python analyze_data.py <root_dir> [--out_dir plots/] [--csv_out summary.csv] [--bins 30]
"""

import os
import re
import argparse
import csv
import math
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure matplotlib has a writable config dir on HPC nodes
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = os.path.join("/tmp", f"matplotlib-{os.getuid()}")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir


# ── Constants ────────────────────────────────────────────────────────────────

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]

# Maps normalised raw keys (from .tgl lines) → canonical event name
_EVENT_KEY_MAP: Dict[str, str] = {
    # asymmetree format
    "speciationfreq":       "Speciation",
    "hgtfreq":              "HGT",
    "lossfreq":             "Loss",
    "duplicationfreq":      "Duplication",
    # old treeducken format
    "cospeciations":        "Speciation",
    "cospeciation":         "Speciation",
    "hostspread/switches":  "HGT",
    "hostspreads/switches": "HGT",
    "hostspread":           "HGT",
    "switches":             "HGT",
    "hostswitches":         "HGT",
}


# ── NEXUS / .tgl parsing ─────────────────────────────────────────────────────

_ALIGN_BLOCK_RE = re.compile(
    r"ALIGNMENT\s*\*\s*(?P<label>\w+)\s*=\s*'(?P<body>.*?)'",
    re.DOTALL | re.IGNORECASE,
)
_KEYVAL_RE = re.compile(
    r"^(?P<key>[A-Za-z0-9_./\- ]+?)\s*[:=,\t ]\s*"
    r"(?P<val>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|None|none)$"
)


def _norm_key(raw: str) -> str:
    """Lower-case, strip spaces and underscores for fuzzy key matching."""
    return raw.strip().lower().replace(" ", "").replace("_", "")


def _parse_align_block(body: str):
    """Return (n_taxa, seq_len) for one ALIGNMENT block."""
    n, seq_len = 0, None
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if seq_len is None:
            seq_len = len(parts[-1])
        n += 1
    return n, (seq_len or 0)


def _scan_tgl(path: str) -> dict:
    """Parse a .tgl / .nex / .nexus file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    result = {
        "host_taxa":    0,
        "host_len":     0,
        "parasite_taxa": 0,
        "parasite_len": 0,
        "events": {e: float("nan") for e in EVENT_NAMES},
    }

    # ── alignment blocks → tree structure ───────────────────────────────
    for label, body in _ALIGN_BLOCK_RE.findall(text):
        n, L = _parse_align_block(body)
        lab = label.lower()
        if "host" in lab:
            result["host_taxa"], result["host_len"] = n, L
        elif "para" in lab or "symbiont" in lab:
            result["parasite_taxa"], result["parasite_len"] = n, L
        else:
            if result["host_taxa"] == 0:
                result["host_taxa"], result["host_len"] = n, L
            else:
                result["parasite_taxa"], result["parasite_len"] = n, L

    # ── line-by-line key/value scan → event frequencies ─────────────────
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _KEYVAL_RE.match(line)
        if m:
            canonical = _EVENT_KEY_MAP.get(_norm_key(m.group("key")))
            if canonical:
                val_str = m.group("val")
                try:
                    result["events"][canonical] = (
                        float("nan") if val_str.lower() in ("nan", "none")
                        else float(val_str)
                    )
                except ValueError:
                    pass
            continue

        # fallback: "Key value" two-token lines (e.g. "Cospeciations 0.42")
        parts = line.split()
        if len(parts) == 2:
            canonical = _EVENT_KEY_MAP.get(_norm_key(parts[0]))
            if canonical:
                try:
                    result["events"][canonical] = float(parts[1].replace(",", "."))
                except ValueError:
                    pass

    return result


def _scan_pt(path: str) -> dict:
    """Parse a pre-encoded .pt file produced by data.py."""
    import torch
    sample = torch.load(path, map_location="cpu", weights_only=False)

    def _msa_shape(dict_key: str, tensor_keys: tuple[str, ...]) -> tuple[int, int]:
        msas = sample.get(dict_key)
        if isinstance(msas, dict):
            first = next(iter(msas.values()), None)
            if first is None:
                return 0, 0
            return (
                len(msas),
                len(first) if isinstance(first, str) else int(first.shape[-1]),
            )
        for key in tensor_keys:
            tensor = sample.get(key)
            if tensor is not None:
                return int(tensor.shape[0]), int(tensor.shape[-1])
        return 0, 0

    host_taxa, host_len = _msa_shape("host_msas", ("host_msa",))
    para_taxa, para_len = _msa_shape("parasite_msas", ("parasite_msa", "para_msa"))
    freqs = sample.get("event_frequencies") or sample.get("labels") or {}

    events: Dict[str, float] = {}
    for e in EVENT_NAMES:
        # Try canonical name, then raw asymmetree key, then old treeducken key
        v = freqs.get(e,
            freqs.get(f"{e}_freq",
            freqs.get({"Speciation": "Cospeciations",
                        "HGT": "Host_spread/Switches"}.get(e, ""), float("nan"))))
        events[e] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")

    return {
        "host_taxa":    host_taxa,
        "host_len":     host_len,
        "parasite_taxa": para_taxa,
        "parasite_len": para_len,
        "events":       events,
    }


def scan_file(path: str) -> dict:
    if path.lower().endswith(".pt"):
        return _scan_pt(path)
    return _scan_tgl(path)


# ── Statistics ───────────────────────────────────────────────────────────────

def summarize(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "count":  int(arr.size),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
    }


def _print_stats(label: str, stats: dict, suffix: str = ""):
    if not stats:
        print(f"  {label}: no data{suffix}")
        return
    print(
        f"  {label}{suffix}: "
        f"n={stats['count']:,}  "
        f"min={stats['min']:.4g}  max={stats['max']:.4g}  "
        f"mean={stats['mean']:.4g}  median={stats['median']:.4g}  "
        f"p90={stats['p90']:.4g}  p95={stats['p95']:.4g}  p99={stats['p99']:.4g}"
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def _save_hist(
    data: np.ndarray,
    title: str,
    xlabel: str,
    outfile: str,
    bins: int = 30,
    dpi: int = 150,
):
    arr = data[np.isfinite(data)]
    if arr.size == 0:
        return
    if np.all(arr == np.round(arr)) and (arr.max() - arr.min()) <= 200:
        bins = np.arange(arr.min() - 0.5, arr.max() + 1.5, 1)
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins, edgecolor="white", linewidth=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()
    print(f"  → {outfile}")


def _save_combined_event_hist(
    event_vals: Dict[str, np.ndarray],
    out_dir: str,
    bins: int = 30,
    dpi: int = 150,
):
    """4-panel figure with all event frequency distributions side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, (e, c) in zip(axes, zip(EVENT_NAMES, colors)):
        arr = event_vals[e]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_title(f"{e} (no data)")
            continue
        ax.hist(arr, bins=bins, color=c, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{e}  (n={arr.size:,})", fontsize=11)
        ax.set_xlabel("frequency")
        ax.set_ylabel("count")
        mean_v = arr.mean()
        ax.axvline(mean_v, color="black", linestyle="--", linewidth=1,
                   label=f"mean={mean_v:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Event frequency distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outfile = os.path.join(out_dir, "event_frequencies_combined.png")
    plt.savefig(outfile, dpi=dpi)
    plt.close()
    print(f"  → {outfile}")


def _save_combined_tree_hist(
    host_taxa: np.ndarray,
    para_taxa: np.ndarray,
    host_len: np.ndarray,
    para_len: np.ndarray,
    out_dir: str,
    bins: int = 30,
    dpi: int = 150,
):
    """4-panel figure with tree structure distributions."""
    datasets = [
        (host_taxa,  "Host leaves per file",        "# leaves"),
        (para_taxa,  "Parasite leaves per file",     "# leaves"),
        (host_len,   "Host alignment length",        "sequence length"),
        (para_len,   "Parasite alignment length",    "sequence length"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (arr, title, xlabel) in zip(axes, datasets):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_title(f"{title} (no data)")
            continue
        is_int = np.all(arr == np.round(arr))
        span = arr.max() - arr.min()
        if is_int and span <= 200:
            hist_bins = np.arange(arr.min() - 0.5, arr.max() + 1.5, 1)
        else:
            hist_bins = bins
        ax.hist(arr, bins=hist_bins, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{title}  (n={arr.size:,})", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean={arr.mean():.1f}")
        ax.legend(fontsize=8)

    fig.suptitle("Tree structure distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outfile = os.path.join(out_dir, "tree_structure_combined.png")
    plt.savefig(outfile, dpi=dpi)
    plt.close()
    print(f"  → {outfile}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analyze Co-Phyloformer datasets: tree sizes and event-frequency "
            "distributions (Speciation, HGT, Loss, Duplication)."
        )
    )
    ap.add_argument("root", help="Directory to scan recursively")
    ap.add_argument(
        "--exts", nargs="+",
        default=[".tgl", ".nex", ".nexus", ".pt"],
        help="File extensions to include (default: .tgl .nex .nexus .pt)",
    )
    ap.add_argument(
        "--out_dir", default="analysis_output",
        help="Output directory for plots (default: analysis_output)",
    )
    ap.add_argument("--bins", type=int, default=30, help="Histogram bins (default: 30)")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Parallel worker threads (default: half of CPUs)",
    )
    ap.add_argument(
        "--csv_out", default=None,
        help="Optional path to write a per-file CSV with all parsed values",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Collect files ────────────────────────────────────────────────────
    files: List[str] = []
    for root_dir, _, fnames in os.walk(args.root):
        for fname in fnames:
            if any(fname.lower().endswith(e.lower()) for e in args.exts):
                files.append(os.path.join(root_dir, fname))

    if not files:
        raise SystemExit(f"No matching files found in: {args.root}")
    print(f"Found {len(files):,} files — parsing with {args.workers} workers …")

    # ── Parse in parallel ────────────────────────────────────────────────
    rows: List[dict] = []
    errors = 0

    def _task(path):
        try:
            r = scan_file(path)
            r["path"] = path
            return r, None
        except Exception as exc:
            return None, (path, str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, p) for p in files]
        for fut in as_completed(futs):
            r, err = fut.result()
            if r is not None:
                rows.append(r)
            else:
                errors += 1

    print(f"Parsed {len(rows):,} files ({errors} errors).\n")

    # ── Build arrays ─────────────────────────────────────────────────────
    host_taxa  = np.array([r["host_taxa"]     for r in rows], dtype=float)
    para_taxa  = np.array([r["parasite_taxa"] for r in rows], dtype=float)
    host_len   = np.array([r["host_len"]      for r in rows], dtype=float)
    para_len   = np.array([r["parasite_len"]  for r in rows], dtype=float)

    # Zero counts mean the block was missing — treat as NaN for stats
    host_taxa[host_taxa == 0] = float("nan")
    para_taxa[para_taxa == 0] = float("nan")
    host_len[host_len   == 0] = float("nan")
    para_len[para_len   == 0] = float("nan")

    event_vals = {
        e: np.array([r["events"][e] for r in rows], dtype=float)
        for e in EVENT_NAMES
    }

    # ── Print summary ────────────────────────────────────────────────────
    print("── Tree structure ──────────────────────────────────────────────")
    _print_stats("Host leaves   ", summarize(host_taxa))
    _print_stats("Parasite leaves", summarize(para_taxa))
    _print_stats("Host seq length ", summarize(host_len))
    _print_stats("Para seq length ", summarize(para_len))

    print("\n── Event frequency distributions ───────────────────────────────")
    for e in EVENT_NAMES:
        arr     = event_vals[e]
        finite  = arr[np.isfinite(arr)]
        missing = int(np.sum(~np.isfinite(arr)))
        suffix  = f"  [{missing:,} missing]" if missing else ""
        _print_stats(f"{e:<14}", summarize(finite), suffix)

    # ── Combined plots ────────────────────────────────────────────────────
    print("\n── Saving plots ────────────────────────────────────────────────")
    _save_combined_tree_hist(host_taxa, para_taxa, host_len, para_len,
                             args.out_dir, args.bins)
    _save_combined_event_hist(event_vals, args.out_dir, args.bins)

    # Individual plots (one per metric)
    _save_hist(host_taxa, "Host leaves per file",     "# leaves",
               os.path.join(args.out_dir, "host_taxa.png"),        args.bins)
    _save_hist(para_taxa, "Parasite leaves per file", "# leaves",
               os.path.join(args.out_dir, "parasite_taxa.png"),    args.bins)
    _save_hist(host_len,  "Host alignment length",    "sequence length",
               os.path.join(args.out_dir, "host_seq_len.png"),     args.bins)
    _save_hist(para_len,  "Parasite alignment length","sequence length",
               os.path.join(args.out_dir, "parasite_seq_len.png"), args.bins)
    for e in EVENT_NAMES:
        _save_hist(event_vals[e], f"{e} frequency distribution", "frequency",
                   os.path.join(args.out_dir, f"{e.lower()}_freq.png"), args.bins)

    # ── Optional per-file CSV ─────────────────────────────────────────────
    if args.csv_out:
        fieldnames = (
            ["path", "host_taxa", "host_len", "parasite_taxa", "parasite_len"]
            + [f"{e}_freq" for e in EVENT_NAMES]
        )
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "path":         r["path"],
                    "host_taxa":    r["host_taxa"],
                    "host_len":     r["host_len"],
                    "parasite_taxa": r["parasite_taxa"],
                    "parasite_len": r["parasite_len"],
                    **{f"{e}_freq": r["events"].get(e, float("nan")) for e in EVENT_NAMES},
                })
        print(f"\nPer-file CSV → {args.csv_out}")


if __name__ == "__main__":
    main()
