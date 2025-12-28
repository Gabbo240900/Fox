#!/usr/bin/env python3
import os, re, argparse, numpy as np, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

ALIGN_BLOCK_RE = re.compile(
    r"ALIGNMENT\s*\*\s*(?P<label>\w+)\s*=\s*'(?P<body>.*?)'",
    re.DOTALL | re.IGNORECASE,
)

def parse_align_block(body: str):
    """
    Parse lines inside an ALIGNMENT block:
        H1  SEQUENCE
        H2  SEQUENCE
    Returns (n_taxa, seq_len) where seq_len is inferred from first non-empty line.
    """
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
    """
    Returns dict with keys:
      host_taxa, host_len, parasite_taxa, parasite_len
    Missing blocks yield 0.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Find all ALIGNMENT blocks
    blocks = ALIGN_BLOCK_RE.findall(text)
    host_taxa = host_len = parasite_taxa = parasite_len = 0

    for label, body in blocks:
        # heuristic: labels like Host1, Para1, HOST, PARASITE
        lab_low = label.lower()
        n, L = parse_align_block(body)
        if "host" in lab_low:
            host_taxa, host_len = n, L
        elif "para" in lab_low or "symbiont" in lab_low:
            parasite_taxa, parasite_len = n, L
        else:
            # fallback: if we only see one block, treat as host
            if host_taxa == 0:
                host_taxa, host_len = n, L
            else:
                parasite_taxa, parasite_len = n, L

    return {
        "host_taxa": host_taxa,
        "host_len": host_len,
        "parasite_taxa": parasite_taxa,
        "parasite_len": parasite_len,
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

def main():
    ap = argparse.ArgumentParser(
        description="Compute taxa-count and alignment-length stats from .tgl/.nex NEXUS files."
    )
    ap.add_argument("root", help="Folder to scan recursively")
    ap.add_argument("--exts", nargs="+", default=[".tgl", ".nex", ".nexus"],
                    help="File extensions to include (default: .tgl .nex .nexus)")
    ap.add_argument("--csv_out", default=None, help="Optional path to write per-file CSV")
    ap.add_argument("--outliers_csv", default=None, help="Optional path to write p95 outlier files (host/parasite)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="Number of threads for parallel parsing (default: half of CPUs)")
    ap.add_argument("--min_taxa", type=int, default=10, help="Minimum taxa required in an alignment block (default: 10)")
    ap.add_argument("--remove", action="store_true", help="Remove files outside thresholds (host/parasite)")
    args = ap.parse_args()

    # Collect all candidate files first
    files_to_scan = []
    for root_dir, _, files in os.walk(args.root):
        for fname in files:
            if any(fname.lower().endswith(e.lower()) for e in args.exts):
                files_to_scan.append(os.path.join(root_dir, fname))

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
                print(f"[WARN] Failed to parse {err[0]}: {err[1]}")

    if not rows:
        print("No matching files found.")
        return

    host_taxa = [r["host_taxa"] for r in rows if r["host_taxa"] > 0]
    para_taxa = [r["parasite_taxa"] for r in rows if r["parasite_taxa"] > 0]
    host_len  = [r["host_len"] for r in rows if r["host_len"] > 0]
    para_len  = [r["parasite_len"] for r in rows if r["parasite_len"] > 0]

    # Percentiles and outliers
    p90_host = int(np.percentile(host_taxa, 90)) if host_taxa else 0
    p90_para = int(np.percentile(para_taxa, 90)) if para_taxa else 0
    min_taxa = int(args.min_taxa)

    outliers_host_hi = [r for r in rows if r["host_taxa"] > p90_host]
    outliers_para_hi = [r for r in rows if r["parasite_taxa"] > p90_para]

    # Low outliers: has a parsed block (>0) but too few taxa
    outliers_host_lo = [r for r in rows if 0 < r["host_taxa"] < min_taxa]
    outliers_para_lo = [r for r in rows if 0 < r["parasite_taxa"] < min_taxa]

    print("\n📦 Files parsed:", len(rows))
    host_stats = summarize(host_taxa)
    para_stats = summarize(para_taxa)
    print("📈 Host taxa stats:", host_stats)
    print("📈 Parasite taxa stats:", para_stats)
    print("🧬 Host alignment length stats:", summarize(host_len))
    print("🧬 Parasite alignment length stats:", summarize(para_len))

    # Show p90 thresholds and outlier files
    print(f"\n🔎 thresholds — min_taxa={min_taxa} | Host taxa p90={p90_host} | Parasite taxa p90={p90_para}")
    if outliers_host_hi:
        print(f"🚩 Host files over p90 ({len(outliers_host_hi)}):")
        for r in sorted(outliers_host_hi, key=lambda x: x['host_taxa'], reverse=True)[:50]:
            print(f"  {r['host_taxa']:>6}  {r['path']}")
        if len(outliers_host_hi) > 50:
            print(f"  ... and {len(outliers_host_hi) - 50} more")
    else:
        print("✅ No host files over p90.")

    if outliers_host_lo:
        print(f"🚩 Host files under min_taxa ({min_taxa}) ({len(outliers_host_lo)}):")
        for r in sorted(outliers_host_lo, key=lambda x: x['host_taxa'])[:50]:
            print(f"  {r['host_taxa']:>6}  {r['path']}")
        if len(outliers_host_lo) > 50:
            print(f"  ... and {len(outliers_host_lo) - 50} more")
    else:
        print("✅ No host files under min_taxa.")

    if outliers_para_hi:
        print(f"🚩 Parasite files over p90 ({len(outliers_para_hi)}):")
        for r in sorted(outliers_para_hi, key=lambda x: x['parasite_taxa'], reverse=True)[:50]:
            print(f"  {r['parasite_taxa']:>6}  {r['path']}")
        if len(outliers_para_hi) > 50:
            print(f"  ... and {len(outliers_para_hi) - 50} more")
    else:
        print("✅ No parasite files over p90.")

    if outliers_para_lo:
        print(f"🚩 Parasite files under min_taxa ({min_taxa}) ({len(outliers_para_lo)}):")
        for r in sorted(outliers_para_lo, key=lambda x: x['parasite_taxa'])[:50]:
            print(f"  {r['parasite_taxa']:>6}  {r['path']}")
        if len(outliers_para_lo) > 50:
            print(f"  ... and {len(outliers_para_lo) - 50} more")
    else:
        print("✅ No parasite files under min_taxa.")

    # Optional removal of outlier files
    if args.remove:
        to_delete = set()
        for r in outliers_host_hi:
            to_delete.add(r["path"])
        for r in outliers_para_hi:
            to_delete.add(r["path"])
        for r in outliers_host_lo:
            to_delete.add(r["path"])
        for r in outliers_para_lo:
            to_delete.add(r["path"])

        if to_delete:
            print(f"\n🗑️ Removing {len(to_delete)} files outside [min_taxa, p90] taxa thresholds...")
            removed_ok = 0
            failed = 0
            for p in sorted(to_delete):
                try:
                    os.remove(p)
                    removed_ok += 1
                    print(f"  ✅ removed: {p}")
                except Exception as e:
                    failed += 1
                    print(f"  ❌ failed:  {p} — {e}")
            print(f"Done. Removed {removed_ok} file(s); {failed} failed.")
        else:
            print("\n✅ No files to remove outside [min_taxa, p90] thresholds.")

    # Optional outliers CSV
    if args.outliers_csv:
        with open(args.outliers_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "type", "side", "taxa", "min_taxa", "p90_threshold"])
            w.writeheader()
            for r in outliers_host_hi:
                w.writerow({"path": r["path"], "type": "host", "side": "high", "taxa": r["host_taxa"], "min_taxa": min_taxa, "p90_threshold": p90_host})
            for r in outliers_host_lo:
                w.writerow({"path": r["path"], "type": "host", "side": "low", "taxa": r["host_taxa"], "min_taxa": min_taxa, "p90_threshold": p90_host})
            for r in outliers_para_hi:
                w.writerow({"path": r["path"], "type": "parasite", "side": "high", "taxa": r["parasite_taxa"], "min_taxa": min_taxa, "p90_threshold": p90_para})
            for r in outliers_para_lo:
                w.writerow({"path": r["path"], "type": "parasite", "side": "low", "taxa": r["parasite_taxa"], "min_taxa": min_taxa, "p90_threshold": p90_para})
        print(f"🧾 Wrote outlier files to {args.outliers_csv}")

    # Optional CSV
    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path","host_taxa","host_len","parasite_taxa","parasite_len"])
            w.writeheader()
            for r in rows:
                w.writerow({
                    "path": r["path"],
                    "host_taxa": r["host_taxa"],
                    "host_len": r["host_len"],
                    "parasite_taxa": r["parasite_taxa"],
                    "parasite_len": r["parasite_len"],
                })
        print(f"\n📝 Wrote per-file details to {args.csv_out}")

if __name__ == "__main__":
    main()