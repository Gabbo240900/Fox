import os
import sys
import argparse
from glob import glob
import torch
from tqdm import tqdm

# Reuse the inference package's reader + encoding so training and `fox predict`
# can never drift apart (tokens, Jukes-Cantor distances, lost-leaf filtering).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from fox.io import read_tgl  # noqa: E402
from fox.encoding import encode_msa, jukes_cantor_dist  # noqa: E402

LABEL_KEYS = ("Speciation", "HGT", "Loss", "Duplication", "Sim_time")


def preencode(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    tgl_paths = sorted(glob(os.path.join(src_dir, "*.tgl")))
    skipped = 0
    for i, path in enumerate(tqdm(tgl_paths)):
        try:
            # drop_lost: keep only leaves alive at the present (no lost genes,
            # no planted-root P0/H0 row), matching what real data looks like.
            sample = read_tgl(path, drop_lost=True)
        except ValueError as e:
            print(f"[skip] {e}")
            skipped += 1
            continue
        host_msa, para_msa = sample["host_msa"], sample["sym_msa"]
        if len(host_msa) < 2 or len(para_msa) < 2:
            skipped += 1
            continue
        h_idx = {n: j for j, n in enumerate(host_msa)}
        p_idx = {n: j for j, n in enumerate(para_msa)}
        mappings = [(h_idx[h], p_idx[p]) for h, p in sample["mapping"]]
        if not mappings:
            skipped += 1
            continue
        host_tokens = encode_msa(host_msa)
        para_tokens = encode_msa(para_msa)
        out = {
            "host_msa":  host_tokens,
            "para_msa":  para_tokens,
            "host_dist": jukes_cantor_dist(host_tokens),
            "para_dist": jukes_cantor_dist(para_tokens),
            "mappings":  mappings,
            "labels":    {e: sample["labels"].get(e, 0.0) for e in LABEL_KEYS},
        }
        torch.save(out, os.path.join(dst_dir, f"{i:07d}.pt"))
    print(f"Encoded {len(tgl_paths) - skipped}/{len(tgl_paths)} files ({skipped} skipped).")


def main():
    p = argparse.ArgumentParser(description="Pre-encode simulated .tgl datasets into .pt tensors.")
    p.add_argument("--src", required=True, help="Directory with raw .tgl files")
    p.add_argument("--dst", required=True, help="Output directory for .pt files")
    args = p.parse_args()
    preencode(args.src, args.dst)


if __name__ == "__main__":
    main()
