import os
import argparse
import csv
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Constants ────────────────────────────────────────────────────────────────

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]

# Fallback key aliases for .pt files encoded with the old data.py key names
_OLD_KEYS = {
    "Speciation": "Cospeciations",
    "HGT":        "Host_spread/Switches",
}


# ── .pt parsing ──────────────────────────────────────────────────────────────

def scan_pt(path: str) -> dict:
    """Load a pre-encoded .pt file and extract structural + label metadata."""
    sample = torch.load(path, map_location="cpu", weights_only=False)

    host_msas = sample.get("host_msas", {})
    para_msas = sample.get("parasite_msas", {})
    freqs     = sample.get("event_frequencies", {})

    def _seq_len(msas: dict) -> int:
        first = next(iter(msas.values()), None)
        if first is None:
            return 0
        # sequences stored as strings
        if isinstance(first, str):
            return len(first)
        # sequences stored as tensors
        return int(first.shape[-1])

    events: Dict[str, float] = {}
    for e in EVENT_NAMES:
        # Try canonical key → raw asymmetree key (e.g. "Speciation_freq") → old treeducken key
        v = freqs.get(e,
            freqs.get(f"{e}_freq",
            freqs.get(_OLD_KEYS.get(e, ""), float("nan"))))
        if v is None or (isinstance(v, float) and math.isnan(v)):
            events[e] = float("nan")
        else:
            events[e] = float(v)

    return {
        "host_taxa":     len(host_msas),
        "parasite_taxa": len(para_msas),
        "host_len":      _seq_len(host_msas),
        "parasite_len":  _seq_len(para_msas),
        "events":        events,
    }


# ── Filter logic ─────────────────────────────────────────────────────────────

def build_reasons(row: dict, args, p_thresh: Optional[int]) -> List[str]:
    """Return a list of reasons to flag this file. Empty → keep."""
    reasons = []
    ht = row["host_taxa"]
    pt = row["parasite_taxa"]
    hl = row["host_len"]
    pl = row["parasite_len"]

    if args.min_taxa:
        if 0 < ht < args.min_taxa:
            reasons.append(f"host_taxa={ht} < min_taxa={args.min_taxa}")
        if 0 < pt < args.min_taxa:
            reasons.append(f"parasite_taxa={pt} < min_taxa={args.min_taxa}")

    if args.max_taxa:
        if ht > args.max_taxa:
            reasons.append(f"host_taxa={ht} > max_taxa={args.max_taxa}")
        if pt > args.max_taxa:
            reasons.append(f"parasite_taxa={pt} > max_taxa={args.max_taxa}")

    if p_thresh is not None:
        if ht > p_thresh:
            reasons.append(f"host_taxa={ht} > p{args.max_taxa_p:.0f}={p_thresh}")
        if pt > p_thresh:
            reasons.append(f"parasite_taxa={pt} > p{args.max_taxa_p:.0f}={p_thresh}")

    if args.min_seq_len:
        if 0 < hl < args.min_seq_len:
            reasons.append(f"host_len={hl} < min_seq_len={args.min_seq_len}")
        if 0 < pl < args.min_seq_len:
            reasons.append(f"parasite_len={pl} < min_seq_len={args.min_seq_len}")

    if args.max_seq_len:
        if hl > args.max_seq_len:
            reasons.append(f"host_len={hl} > max_seq_len={args.max_seq_len}")
        if pl > args.max_seq_len:
            reasons.append(f"parasite_len={pl} > max_seq_len={args.max_seq_len}")

    if args.no_events:
        if all(math.isnan(row["events"].get(e, float("nan"))) for e in EVENT_NAMES):
            reasons.append("all event frequencies missing")

    return reasons


# ── File operations ───────────────────────────────────────────────────────────

def _unique_dst(dst: str) -> str:
    """Append _1, _2, … before the extension until the path does not exist."""
    if not os.path.exists(dst):
        return dst
    base, ext = os.path.splitext(dst)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def _move(src: str, root: str, side_dir: str) -> bool:
    try:
        rel = os.path.relpath(src, root)
        dst = _unique_dst(os.path.join(side_dir, rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return True
    except Exception as exc:
        print(f"  ✗ move failed: {src}  —  {exc}")
        return False


def _delete(src: str) -> bool:
    try:
        os.remove(src)
        return True
    except Exception as exc:
        print(f"  ✗ delete failed: {src}  —  {exc}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Filter pre-encoded .pt Co-Phyloformer dataset files by size / label quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("root", help="Directory containing .pt files (scanned recursively)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="Parallel load threads (default: half of CPUs)")

    # ── thresholds ──────────────────────────────────────────────────────
    th = ap.add_argument_group("filter thresholds")
    th.add_argument("--min_taxa",    type=int,   default=None,
                    help="Flag files with fewer than N host or parasite leaves")
    th.add_argument("--max_taxa",    type=int,   default=None,
                    help="Flag files with more than N host or parasite leaves")
    th.add_argument("--max_taxa_p",  type=float, default=None,
                    help="Flag files above the P-th percentile of leaf count (e.g. 95)")
    th.add_argument("--min_seq_len", type=int,   default=None,
                    help="Flag files with alignment length shorter than N")
    th.add_argument("--max_seq_len", type=int,   default=None,
                    help="Flag files with alignment length longer than N")
    th.add_argument("--no_events",   action="store_true",
                    help="Flag files where all four event frequencies are missing")

    # ── actions ─────────────────────────────────────────────────────────
    act = ap.add_mutually_exclusive_group()
    act.add_argument("--move",   action="store_true",
                     help="Move flagged files to --side_dir (safe, reversible)")
    act.add_argument("--delete", action="store_true",
                     help="Permanently delete flagged files  ⚠ irreversible")

    ap.add_argument("--side_dir", default=None,
                    help="Destination folder for --move (default: <root>/filtered_out/)")
    ap.add_argument("--csv_out",  default=None,
                    help="Write a CSV listing all flagged files with reasons")
    args = ap.parse_args()

    # ── collect .pt files ────────────────────────────────────────────────
    files: List[str] = []
    for root_dir, _, fnames in os.walk(args.root):
        for fname in fnames:
            if fname.lower().endswith(".pt"):
                files.append(os.path.join(root_dir, fname))

    if not files:
        raise SystemExit(f"No .pt files found in: {args.root}")
    print(f"Found {len(files):,} .pt files — loading with {args.workers} workers …")

    # ── parse in parallel ─────────────────────────────────────────────────
    rows: List[dict] = []
    errors = 0

    def _task(path):
        try:
            r = scan_pt(path)
            r["path"] = path
            return r, None
        except Exception as exc:
            return None, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, p) for p in files]
        for fut in as_completed(futs):
            r, err = fut.result()
            if r is not None:
                rows.append(r)
            else:
                errors += 1

    print(f"Loaded {len(rows):,} files ({errors} errors).\n")

    # ── percentile threshold ──────────────────────────────────────────────
    p_thresh: Optional[int] = None
    if args.max_taxa_p is not None:
        all_taxa = [
            v
            for r in rows
            for v in (r["host_taxa"], r["parasite_taxa"])
            if v > 0
        ]
        if all_taxa:
            p_thresh = int(np.percentile(all_taxa, args.max_taxa_p))
            print(f"p{args.max_taxa_p:.0f} leaf threshold: {p_thresh} leaves\n")

    # ── apply filters ─────────────────────────────────────────────────────
    flagged: List[Tuple[dict, List[str]]] = []
    for r in rows:
        reasons = build_reasons(r, args, p_thresh)
        if reasons:
            flagged.append((r, reasons))

    kept = len(rows) - len(flagged)
    print(f"Total: {len(rows):,}  │  Flagged: {len(flagged):,}  │  Kept: {kept:,}\n")

    if not flagged:
        print("✅ No files match the filter criteria — nothing to remove.")
        return

    # ── print flagged list ────────────────────────────────────────────────
    print(f"{'FILE':<70}  REASON(S)")
    print("─" * 100)
    for r, reasons in sorted(flagged, key=lambda x: x[0]["path"]):
        print(f"  {r['path']:<68}  {'; '.join(reasons)}")

    # ── optional CSV export ───────────────────────────────────────────────
    if args.csv_out:
        fieldnames = ["path", "host_taxa", "parasite_taxa",
                      "host_len", "parasite_len", "reasons"]
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r, reasons in flagged:
                w.writerow({
                    "path":          r["path"],
                    "host_taxa":     r["host_taxa"],
                    "parasite_taxa": r["parasite_taxa"],
                    "host_len":      r["host_len"],
                    "parasite_len":  r["parasite_len"],
                    "reasons":       "; ".join(reasons),
                })
        print(f"\nFlagged-file list → {args.csv_out}")

    # ── dry run ───────────────────────────────────────────────────────────
    if not args.move and not args.delete:
        print(
            f"\nDry run — {len(flagged):,} file(s) would be affected.\n"
            f"Re-run with --move or --delete to act on them."
        )
        return

    # ── move ──────────────────────────────────────────────────────────────
    if args.move:
        side_dir = args.side_dir or os.path.join(args.root, "filtered_out")
        os.makedirs(side_dir, exist_ok=True)
        print(f"\nMoving {len(flagged):,} file(s) → {side_dir} …")
        ok = failed = 0
        for r, _ in flagged:
            (ok := ok + 1) if _move(r["path"], args.root, side_dir) else (failed := failed + 1)
        print(f"Done. Moved {ok:,}  │  Failed {failed:,}")

    # ── delete ────────────────────────────────────────────────────────────
    if args.delete:
        print(f"\n⚠  Deleting {len(flagged):,} file(s) …")
        ok = failed = 0
        for r, _ in flagged:
            (ok := ok + 1) if _delete(r["path"]) else (failed := failed + 1)
        print(f"Done. Deleted {ok:,}  │  Failed {failed:,}")


if __name__ == "__main__":
    main()
