#!/usr/bin/env python3
"""Inference script for Co-Phyloformer (matches your current train.py pipeline).

It expects NEW DATA in the same *raw pre-encoded* format you train on:
  - sample['host_msas']: dict{name -> sequence_str}
  - sample['parasite_msas']: dict{name -> sequence_str}
  - sample['mappings']: list of (parasite_leaf, host_leaf)
  - sample['event_frequencies']: dict with keys like 'Cospeciations', 'Host_spread/Switches', 'Sim_time'

This script will:
  - encode sequences (same encode_sequence as train.py)
  - build valid_mappings as index pairs
  - pad the number-of-sequences dimension across batch (same as train.py collate_fn)
  - run model(host_msa, parasite_msa, mappings, sim_time)
  - save predictions to CSV

Example
-------
python test.py \
  --checkpoint /path/to/best_overall_val_epoch5_step1234.pth \
  --data_dir /path/to/new_preencoded_pt \
  --out_csv preds.csv \
  --batch_size 16 \
  --device cuda
"""

import argparse
import csv
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


EVENT_NAMES = [
    "Cospeciations",
    "Host_spread/Switches",  # your train.py uses this spelling
]


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def encode_sequence(sequence: str, max_len: int = 128) -> torch.Tensor:
    """Same encoding as train.py."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"  # 21 tokens: 20 AAs + gap
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    UNK_ID = 21
    PAD_ID = 22

    encoded = [aa_to_index.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


class RawPtToModelInputs(Dataset):
    """Loads raw .pt files and converts them to the exact model inputs used in training."""

    def __init__(self, data_dir: str, limit: Optional[int] = None, max_len: int = 128):
        self.pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if limit is not None:
            self.pt_files = self.pt_files[: int(limit)]
        if not self.pt_files:
            raise FileNotFoundError(f"No .pt files found in: {data_dir}")
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pt_path = self.pt_files[idx]
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Skip empty samples (same as train)
        if len(sample.get("host_msas", {})) == 0 or len(sample.get("parasite_msas", {})) == 0:
            return {"__skip__": True, "__file__": os.path.basename(pt_path)}

        host_list = list(sample["host_msas"].keys())
        parasite_list = list(sample["parasite_msas"].keys())

        parasite_idx_map = {name: i for i, name in enumerate(parasite_list)}
        host_idx_map = {name: i for i, name in enumerate(host_list)}

        valid_mappings: List[Tuple[int, int]] = [
            (host_idx_map[h], parasite_idx_map[p])
            for p, h in sample.get("mappings", [])
            if p in parasite_idx_map and h in host_idx_map
        ]

        labels = torch.tensor(
            [sample.get("event_frequencies", {}).get(event, 0.0) for event in EVENT_NAMES],
            dtype=torch.float32,
        )

        sim_time = torch.tensor([
            _safe_float(sample.get("event_frequencies", {}).get("Sim_time", 1.0), default=1.0)
        ], dtype=torch.float32)

        host_msa = torch.stack([encode_sequence(seq, max_len=self.max_len) for seq in sample["host_msas"].values()])
        parasite_msa = torch.stack([encode_sequence(seq, max_len=self.max_len) for seq in sample["parasite_msas"].values()])

        return {
            "__file__": os.path.basename(pt_path),
            "host_msa": host_msa,
            "parasite_msa": parasite_msa,
            "mappings": valid_mappings,
            "labels": labels,
            "sim_time": sim_time,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Same padding logic as train.py: pad number-of-sequences dimension across batch."""
    batch = [b for b in batch if not b.get("__skip__", False)]
    if len(batch) == 0:
        return None

    files = [b["__file__"] for b in batch]
    host_msas = [b["host_msa"] for b in batch]
    parasite_msas = [b["parasite_msa"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    mappings = [b["mappings"] for b in batch]
    sim_time = torch.stack([b["sim_time"] for b in batch])

    max_host_len = max(m.shape[0] for m in host_msas)
    max_parasite_len = max(m.shape[0] for m in parasite_msas)

    # F.pad pads last dims; here tensors are [num_seqs, seq_len]
    host_msas = [F.pad(m, (0, 0, 0, max_host_len - m.shape[0]), value=22) for m in host_msas]
    parasite_msas = [F.pad(m, (0, 0, 0, max_parasite_len - m.shape[0]), value=22) for m in parasite_msas]

    host_msas = torch.stack(host_msas)
    parasite_msas = torch.stack(parasite_msas)

    return {
        "__file__": files,
        "host_msa": host_msas,
        "parasite_msa": parasite_msas,
        "labels": labels,
        "mappings": mappings,
        "sim_time": sim_time,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Co-Phyloformer inference on raw pre-encoded .pt samples")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing .pt samples")
    p.add_argument("--out_csv", type=str, default="predictions.csv", help="Output CSV path")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 etc. (default: auto)")
    p.add_argument("--limit", type=int, default=None, help="Optionally limit number of samples")
    p.add_argument("--max_len", type=int, default=128, help="Sequence length used in encoding")
    return p.parse_args()


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    from model import Cophyloformer  # matches your train.py

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt

    if state_dict is None:
        raise ValueError(
            "Could not find a model state_dict in the checkpoint. "
            "Expected keys like 'model_state_dict' or 'state_dict'."
        )

    model = Cophyloformer()

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading state_dict ({len(missing)}). First few: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading state_dict ({len(unexpected)}). First few: {unexpected[:10]}")

    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ds = RawPtToModelInputs(args.data_dir, limit=args.limit, max_len=args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            files = batch["__file__"]
            host_msa = batch["host_msa"].to(device, non_blocking=True)
            parasite_msa = batch["parasite_msa"].to(device, non_blocking=True)
            sim_time = batch["sim_time"].to(device, non_blocking=True)
            labels = batch["labels"]  # keep on CPU for CSV
            mappings = batch["mappings"]

            outputs = model(host_msa, parasite_msa, mappings, sim_time)
            preds = outputs.detach().float().cpu()

            for i in range(preds.shape[0]):
                row = {
                    "file": files[i],
                    "pred_Cospeciations": float(preds[i, 0]),
                    "pred_Host_spread/Switches": float(preds[i, 1]) if preds.shape[1] > 1 else None,
                    "gt_Cospeciations": float(labels[i, 0]),
                    "gt_Host_spread/Switches": float(labels[i, 1]) if labels.shape[1] > 1 else None,
                }
                rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["file", "pred_Cospeciations", "gt_Cospeciations", "pred_Host_spread/Switches", "gt_Host_spread/Switches"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {args.out_csv}")


if __name__ == "__main__":
    main()