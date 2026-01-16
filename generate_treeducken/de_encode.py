#!/usr/bin/env python3
import os
import glob
import argparse
import torch

def _pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _as_newick(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip()
    # sometimes stored as bytes
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8").strip()
    return str(x).strip()

def _normalize_associations(assoc):
    """
    Return list of (host_leaf, parasite_leaf).
    Supports:
      - list of pairs: [("H1","P1"), ...] or [["H1","P1"], ...]
      - list of dicts: [{"host_leaf":"H1","parasite_leaf":"P1"}, ...]
      - dict mapping: {"H1":"P1", "H2":"P9", ...}
    """
    if assoc is None:
        return []

    pairs = []

    if isinstance(assoc, dict):
        # could be {"H1":"P1"} or {"H1":["P1","P2"]}
        for h, p in assoc.items():
            if isinstance(p, (list, tuple)):
                for pi in p:
                    pairs.append((str(h), str(pi)))
            else:
                pairs.append((str(h), str(p)))
        return pairs

    if isinstance(assoc, (list, tuple)):
        for item in assoc:
            if isinstance(item, dict):
                h = item.get("host_leaf") or item.get("host") or item.get("h")
                p = item.get("parasite_leaf") or item.get("parasite") or item.get("p")
                if h is not None and p is not None:
                    pairs.append((str(h), str(p)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
        return pairs

    return []

def write_tgl(out_path, host_newick, parasite_newick, associations, event_freqs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# HOST_TREE\n")
        f.write((host_newick or "") + ("\n" if (host_newick or "").endswith("\n") else "\n"))
        f.write("\n# PARASITE_TREE\n")
        f.write((parasite_newick or "") + ("\n" if (parasite_newick or "").endswith("\n") else "\n"))

        f.write("\n# ASSOCIATIONS (host, parasite)\n")
        for h, p in associations:
            f.write(f"{h},{p}\n")

        if isinstance(event_freqs, dict) and len(event_freqs) > 0:
            f.write("\n# EVENT_FREQUENCIES\n")
            for k, v in event_freqs.items():
                f.write(f"{k}:{v}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing .pt files")
    ap.add_argument("--out_dir", required=True, help="Output folder for .tgl files")
    args = ap.parse_args()

    pt_files = sorted(glob.glob(os.path.join(args.in_dir, "*.pt")))
    if not pt_files:
        raise SystemExit(f"No .pt files found in: {args.in_dir}")

    n_ok = 0
    n_skip = 0

    for pt in pt_files:
        try:
            sample = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[SKIP] {pt} (torch.load failed: {e})")
            n_skip += 1
            continue

        host_newick = _as_newick(_pick(sample, ["host_newick", "host_tree_newick", "host_tree"]))
        parasite_newick = _as_newick(_pick(sample, ["parasite_newick", "parasite_tree_newick", "parasite_tree"]))

        assoc_raw = _pick(sample, ["associations", "mapping", "host_parasite_mapping", "links"])
        associations = _normalize_associations(assoc_raw)

        event_freqs = _pick(sample, ["event_frequencies", "events", "rates"])

        # Hard requirement to rebuild a meaningful .tgl:
        if not host_newick or not parasite_newick or len(associations) == 0:
            print(f"[SKIP] {os.path.basename(pt)} missing host/parasite/mapping in .pt")
            n_skip += 1
            continue

        base = os.path.splitext(os.path.basename(pt))[0]
        out_path = os.path.join(args.out_dir, base + ".tgl")
        write_tgl(out_path, host_newick, parasite_newick, associations, event_freqs)
        n_ok += 1

    print(f"\nDone. Wrote {n_ok} .tgl files. Skipped {n_skip}.")

if __name__ == "__main__":
    main()