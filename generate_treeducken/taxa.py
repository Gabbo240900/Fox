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
    ap.add_argument("--remove", action="store_true", help="Remove files above the p99 taxa thresholds (host/parasite)")
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
    p99_host = int(np.percentile(host_taxa, 99)) if host_taxa else 0
    p99_para = int(np.percentile(para_taxa, 99)) if para_taxa else 0

    outliers_host = [r for r in rows if r["host_taxa"] > p99_host]
    outliers_para = [r for r in rows if r["parasite_taxa"] > p99_para]

    print("\n📦 Files parsed:", len(rows))
    host_stats = summarize(host_taxa)
    para_stats = summarize(para_taxa)
    print("📈 Host taxa stats:", host_stats)
    print("📈 Parasite taxa stats:", para_stats)
    print("🧬 Host alignment length stats:", summarize(host_len))
    print("🧬 Parasite alignment length stats:", summarize(para_len))

    # Show p99 thresholds and outlier files
    print(f"\n🔎 p99 thresholds — Host taxa: {p99_host}, Parasite taxa: {p99_para}")
    if outliers_host:
        print(f"🚩 Host files over p99 ({len(outliers_host)}):")
        for r in sorted(outliers_host, key=lambda x: x['host_taxa'], reverse=True)[:50]:
            print(f"  {r['host_taxa']:>6}  {r['path']}")
        if len(outliers_host) > 50:
            print(f"  ... and {len(outliers_host) - 50} more")
    else:
        print("✅ No host files over p99.")

    if outliers_para:
        print(f"🚩 Parasite files over p99 ({len(outliers_para)}):")
        for r in sorted(outliers_para, key=lambda x: x['parasite_taxa'], reverse=True)[:50]:
            print(f"  {r['parasite_taxa']:>6}  {r['path']}")
        if len(outliers_para) > 50:
            print(f"  ... and {len(outliers_para) - 50} more")
    else:
        print("✅ No parasite files over p99.")

    # Optional removal of outlier files
    if args.remove:
        to_delete = set()
        for r in outliers_host:
            to_delete.add(r["path"])
        for r in outliers_para:
            to_delete.add(r["path"])

        if to_delete:
            print(f"\n🗑️ Removing {len(to_delete)} files above p99 thresholds...")
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
            print("\n✅ No files to remove above p99 thresholds.")

    # Optional outliers CSV
    if args.outliers_csv:
        with open(args.outliers_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path","type","taxa","p99_threshold"])
            w.writeheader()
            for r in outliers_host:
                w.writerow({"path": r["path"], "type": "host", "taxa": r["host_taxa"], "p99_threshold": p99_host})
            for r in outliers_para:
                w.writerow({"path": r["path"], "type": "parasite", "taxa": r["parasite_taxa"], "p99_threshold": p99_para})
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