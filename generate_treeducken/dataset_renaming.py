from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple

NAME_PATTERN = re.compile(r"^Dataset(\d+)\.(?P<ext>[A-Za-z0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move & renumber Dataset*.EXT files.")
    parser.add_argument(
        "--src",
        default="Dataset",
        type=str,
        help="Source folder containing files to move (default: ./Dataset)",
    )
    parser.add_argument(
        "--dst",
        default="Dataset_final",
        type=str,
        help="Destination folder where files are moved (default: ./Dataset_final)",
    )
    parser.add_argument(
        "--ext",
        default=".tgl",
        type=str,
        help="File extension to match, including dot (default: .tgl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving files",
    )
    return parser.parse_args()


def extract_index_and_ext(path: Path) -> Tuple[int, str] | None:
    """Return (index, ext) if filename matches Dataset<idx>.<ext>; else None."""
    m = NAME_PATTERN.match(path.name)
    if not m:
        return None
    try:
        idx = int(m.group(1))
    except ValueError:
        return None
    return idx, m.group("ext")


def find_max_index(dst_dir: Path, expected_ext: str) -> int:
    max_idx = 0
    for p in dst_dir.iterdir() if dst_dir.exists() else []:
        parsed = extract_index_and_ext(p)
        if parsed is None:
            continue
        idx, ext = parsed
        if expected_ext and ("." + ext.lower()) != expected_ext.lower():
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx


def list_source_files(src_dir: Path, expected_ext: str) -> List[Path]:
    candidates: List[Path] = []
    for p in src_dir.iterdir() if src_dir.exists() else []:
        if not p.is_file():
            continue
        parsed = extract_index_and_ext(p)
        if parsed is None:
            continue
        idx, ext = parsed
        if expected_ext and ("." + ext.lower()) != expected_ext.lower():
            continue
        candidates.append(p)
    # sort by numeric index
    candidates.sort(key=lambda x: extract_index_and_ext(x)[0])
    return candidates


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    ext = args.ext if args.ext.startswith(".") else "." + args.ext

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source folder not found or not a directory: {src_dir}")

    # Create destination if needed
    if not dst_dir.exists():
        if args.dry_run:
            print(f"[dry-run] Would create destination folder: {dst_dir}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created destination folder: {dst_dir}")

    max_existing = find_max_index(dst_dir, ext)
    print(f"Max existing index in destination: {max_existing}")

    sources = list_source_files(src_dir, ext)
    if not sources:
        print("No matching files found in source. Nothing to do.")
        return

    next_idx = max_existing + 1
    moves: List[Tuple[Path, Path]] = []

    for src in sources:
        dest_name = f"Dataset{next_idx}{ext}"
        dst_path = dst_dir / dest_name
        # Safety: don't overwrite unexpectedly
        if dst_path.exists():
            raise SystemExit(
                f"Refusing to overwrite existing file: {dst_path}. Aborting."
            )
        moves.append((src, dst_path))
        next_idx += 1

    # Show plan
    print(f"Planned moves ({len(moves)} files):")
    for s, d in moves:
        print(f"  {s.name}  =>  {d.name}")

    if args.dry_run:
        print("[dry-run] No files moved.")
        return

    # Execute moves
    for s, d in moves:
        shutil.move(str(s), str(d))

    print("Done.")


if __name__ == "__main__":
    main()