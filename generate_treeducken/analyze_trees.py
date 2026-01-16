#!/usr/bin/env python3
import os, re, argparse, numpy as np, csv, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

ALIGN_BLOCK_RE = re.compile(
    r"ALIGNMENT\s*\*\s*(?P<label>\w+)\s*=\s*'(?P<body>.*?)'",
    re.DOTALL | re.IGNORECASE,
)

COSP_RE = re.compile(r"^Cospeciations\s+([\-\d.eE+,]+|NaN)$", re.IGNORECASE | re.MULTILINE)

def parse_align_block(body: str):
    taxa = []
    seq_len = None
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        # Expect "<id><whitespace><sequence>"
        parts = line.split()
        if len(parts) < 2:
            continue
        seq = parts[-1]
        if seq_len is None:
            seq_len = len(seq)
        taxa.append(1)
    return len(taxa), (seq_len or 0)

def scan_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

        # Extract Cospeciations (if present)
        cospeciation = None
        m = COSP_RE.search(text)
        if m:
            val_str = m.group(1).strip()
            if val_str.lower() != "nan":
                try:
                    cospeciation = float(val_str.replace(",", "."))
                except ValueError:
                    cospeciation = None

    # Find all ALIGNMENT blocks
    blocks = ALIGN_BLOCK_RE.findall(text)
    host_taxa = parasite_taxa = 0

    for label, body in blocks:
        # heuristic: labels like Host1, Para1, HOST, PARASITE
        lab_low = label.lower()
        n, _ = parse_align_block(body)
        if "host" in lab_low:
            host_taxa = n
        elif "para" in lab_low or "symbiont" in lab_low:
            parasite_taxa = n
        else:
            # fallback: if we only see one block, treat as host
            if host_taxa == 0:
                host_taxa = n
            else:
                parasite_taxa = n

    return {
        "host_taxa": host_taxa,
        "parasite_taxa": parasite_taxa,
        "cospeciation": cospeciation,
    }

def summarize(arr):
    arr = np.array(arr, dtype=np.int64)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }

def value_counts(arr):
    """Return a dict mapping unique values to their counts."""
    counts = defaultdict(int)
    for v in arr:
        counts[int(v)] += 1
    return dict(counts)

def print_distribution(title, counts_dict, top_n=25):
    if not counts_dict:
        print(f"\n {title}: (no data)")
        return
    # Sort by value (key) ascending for readability
    items = sorted(counts_dict.items(), key=lambda x: x[0])
    print(f"\n {title} — unique values: {len(items)}")
    print("value -> #files")
    shown = 0
    for val, cnt in items:
        print(f"  {val:>5} -> {cnt}")
        shown += 1
        if shown >= top_n:
            remaining = len(items) - shown
            if remaining > 0:
                print(f"  ... and {remaining} more")
            break

def save_hist(data, title, xlabel, outfile, dpi=150):
    if not data:
        return
    plt.figure()
    plt.hist(data, bins='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def save_bar_from_counts(counts_dict, title, xlabel, outfile, dpi=150):
    if not counts_dict:
        return
    xs = sorted(counts_dict.keys())
    ys = [counts_dict[x] for x in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Files')
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser(
        description="Compute taxa-count stats from .tgl/.nex NEXUS files."
    )
    ap.add_argument("root", help="Folder to scan recursively")
    ap.add_argument("--csv_out", default=None, help="Optional path to write per-file CSV")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="Number of threads for parallel parsing (default: half of CPUs)")
    ap.add_argument("--out_dir", default="analysis_plots", help="Directory where PNGs will be written (default: analysis_plots)")
    ap.add_argument("--extract_extreme_cosp", action="store_true",
                    help="If set, move datasets with Cospeciations exactly 0 or 1 into an extreme_cosp_data folder")
    ap.add_argument("--extreme_dir", default="/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/extreme_cosp_data",
                    help="Destination root for extreme cospeciation datasets (default: Jean-Zay fsn1 path)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all candidate files first
    files_to_scan = []
    for fname in os.listdir(args.root):
        low = fname.lower()
        if low.endswith(".tgl") or low.endswith(".nex"):
            files_to_scan.append(os.path.join(args.root, fname))

    if not files_to_scan:
        print("No matching files found.")
        return

    # Parse in parallel
    rows = []
    def _task(path):
        try:
            r = scan_file(path)
            r["path"] = path
            return r, None
        except Exception as e:
            return None, (path, str(e))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_task, p) for p in files_to_scan]
        for fut in as_completed(futures):
            r, err = fut.result()
            if r is not None:
                rows.append(r)
            else:
                pass  # no stdout printing requested

    if not rows:
        return

    # Optionally move extreme cospeciation datasets
    if args.extract_extreme_cosp:
        c0_dir = os.path.join(args.extreme_dir, "cosp_0")
        c1_dir = os.path.join(args.extreme_dir, "cosp_1")
        os.makedirs(c0_dir, exist_ok=True)
        os.makedirs(c1_dir, exist_ok=True)

        moved0 = moved1 = 0
        for r in rows:
            cosp = r.get("cospeciation", None)
            src = r.get("path")
            if src is None or cosp is None:
                continue

            # Exact equality as requested
            if cosp == 0.0:
                dst = os.path.join(c0_dir, os.path.basename(src))
                try:
                    shutil.move(src, dst)
                    moved0 += 1
                    r["path"] = dst
                except Exception:
                    # Fallback: copy + remove
                    shutil.copy2(src, dst)
                    os.remove(src)
                    moved0 += 1
                    r["path"] = dst
            elif cosp == 1.0:
                dst = os.path.join(c1_dir, os.path.basename(src))
                try:
                    shutil.move(src, dst)
                    moved1 += 1
                    r["path"] = dst
                except Exception:
                    shutil.copy2(src, dst)
                    os.remove(src)
                    moved1 += 1
                    r["path"] = dst

        print(f"Moved extreme cospeciation datasets: cosp_0={moved0}, cosp_1={moved1} -> {args.extreme_dir}")

    host_taxa = [r["host_taxa"] for r in rows if r["host_taxa"] > 0]
    para_taxa = [r["parasite_taxa"] for r in rows if r["parasite_taxa"] > 0]

    # Save plots (no printing)
    host_counts = value_counts(host_taxa)
    para_counts = value_counts(para_taxa)

    # Histograms
    save_hist(host_taxa, "Host leaves per file (histogram)", "# host leaves", os.path.join(args.out_dir, f"host_leaves_hist.png"), dpi=150)
    save_hist(para_taxa, "Parasite leaves per file (histogram)", "# parasite leaves", os.path.join(args.out_dir, f"parasite_leaves_hist.png"), dpi=150)

    # Discrete distributions as bar charts
    save_bar_from_counts(host_counts, "Host leaves per file (distribution)", "# host leaves", os.path.join(args.out_dir, f"host_leaves_dist.png"), dpi=150)
    save_bar_from_counts(para_counts, "Parasite leaves per file (distribution)", "# parasite leaves", os.path.join(args.out_dir, f"parasite_leaves_dist.png"), dpi=150)

    # Optional CSV: summaries only (no per-file rows)
    if args.csv_out:
        host_summary = summarize(host_taxa)
        para_summary = summarize(para_taxa)
        fields = ["set","count","min","max","mean","median","p90","p95","p99"]
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            if host_summary:
                w.writerow({"set": "host", **host_summary})
            if para_summary:
                w.writerow({"set": "parasite", **para_summary})

if __name__ == "__main__":
    main()