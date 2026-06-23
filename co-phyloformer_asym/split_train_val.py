"""Split pre-encoded .pt samples into train/val dirs by moving files.
Deterministic (seeded). Run AFTER filtering, BEFORE bucketing."""
import argparse, glob, os, random, shutil


def main():
    ap = argparse.ArgumentParser(description="Move .pt samples into train/val dirs")
    ap.add_argument("--src", required=True, help="Dir with filtered .pt files")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction to val (default 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.pt")))
    if not files:
        raise SystemExit(f"No .pt files in {args.src}")

    random.Random(args.seed).shuffle(files)
    n_val = int(round(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)

    for f in val_files:
        shutil.move(f, os.path.join(args.val_dir, os.path.basename(f)))
    for f in train_files:
        shutil.move(f, os.path.join(args.train_dir, os.path.basename(f)))

    print(f"Split {len(files)} files -> train={len(train_files)}  val={len(val_files)} "
          f"(val_frac={args.val_frac}, seed={args.seed})")


if __name__ == "__main__":
    main()
