import argparse
import csv
import glob
import os

import torch
from tqdm import tqdm


BUCKET_META_NAME = "bucket_meta.tsv"


def write_bucket_metadata(preencoded_dir: str) -> str:
    pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {preencoded_dir}")

    out_path = os.path.join(preencoded_dir, BUCKET_META_NAME)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"

    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file", "host_leaves", "para_leaves", "mapping_count"])
        for pt_path in tqdm(pt_files, desc=f"Buckets {os.path.basename(os.path.normpath(preencoded_dir))}"):
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
            writer.writerow([
                os.path.basename(pt_path),
                int(sample["host_msa"].shape[0]),
                int(sample["para_msa"].shape[0]),
                int(len(sample["mappings"])),
            ])

    os.replace(tmp_path, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build bucket metadata for preencoded Prophet samples")
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more preencoded directories containing .pt samples",
    )
    args = parser.parse_args()

    for preencoded_dir in args.dirs:
        out_path = write_bucket_metadata(preencoded_dir)
        print(f"[Buckets] wrote {out_path}")


if __name__ == "__main__":
    main()
