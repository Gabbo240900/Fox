import argparse
import csv
import glob
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Cophyloformer

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID, PAD_ID = 21, 22


def encode_sequence(sequence, max_len=128):
    encoded = [AA_TO_INDEX.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


def jukes_cantor_dist(msa: torch.Tensor) -> torch.Tensor:
    valid = (msa != PAD_ID)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(0)
    n_v = valid_pair.sum(dim=2).clamp(min=1).float()
    mismatch = (msa.unsqueeze(1) != msa.unsqueeze(0)) & valid_pair
    p = (mismatch.float().sum(dim=2) / n_v).clamp(0.0, 0.74)
    dist = -0.75 * torch.log(1.0 - (4.0 / 3.0) * p)
    dist.fill_diagonal_(0.0)
    return dist


class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir, pt_files=None):
        if pt_files is None:
            self.pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
        else:
            self.pt_files = list(pt_files)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx], map_location="cpu", weights_only=False)

        if not sample.get("host_msas") or not sample.get("parasite_msas"):
            return None

        host_list = list(sample["host_msas"].keys())
        para_list = list(sample["parasite_msas"].keys())
        h_idx = {n: i for i, n in enumerate(host_list)}
        p_idx = {n: i for i, n in enumerate(para_list)}
        mappings = [
            (h_idx[h], p_idx[p])
            for p, h in sample["mappings"]
            if p in p_idx and h in h_idx
        ]

        out = {
            "host_msa": torch.stack([encode_sequence(s) for s in sample["host_msas"].values()]),
            "parasite_msa": torch.stack([encode_sequence(s) for s in sample["parasite_msas"].values()]),
            "mappings": mappings,
            "labels": torch.tensor(
                [sample["event_frequencies"].get(e, 0.0) for e in EVENT_NAMES],
                dtype=torch.float32,
            ),
            "sim_time": torch.tensor(
                [sample["event_frequencies"].get("Sim_time", 1.0)],
                dtype=torch.float32,
            ),
        }
        if "host_dist" in sample:
            out["host_dist"] = sample["host_dist"]
            out["para_dist"] = sample["para_dist"]
        return out


def make_collate_fn(host_max_leaves, para_max_leaves, use_dist_matrix):
    def collate_fn(batch):
        batch = [s for s in batch if s is not None]
        if len(batch) == 0:
            return None

        def pad(msas, cap, pad_val=PAD_ID):
            max_n = min(max(m.shape[0] for m in msas), cap)
            return torch.stack(
                [F.pad(m[:max_n], (0, 0, 0, max(0, max_n - m.shape[0])), value=pad_val) for m in msas]
            )

        host_msas = pad([s["host_msa"] for s in batch], host_max_leaves)
        para_msas = pad([s["parasite_msa"] for s in batch], para_max_leaves)
        out = {
            "host_msa": host_msas,
            "parasite_msa": para_msas,
            "labels": torch.stack([s["labels"] for s in batch]),
            "mappings": [s["mappings"] for s in batch],
            "sim_time": torch.stack([s["sim_time"] for s in batch]),
        }

        if use_dist_matrix:
            if "host_dist" in batch[0]:
                def pad_dist(d, n):
                    if d.shape[0] == n:
                        return d
                    padded = torch.zeros(n, n, dtype=d.dtype)
                    padded[:d.shape[0], :d.shape[0]] = d
                    return padded

                out["host_dist"] = torch.stack([pad_dist(s["host_dist"], host_msas.shape[1]) for s in batch])
                out["para_dist"] = torch.stack([pad_dist(s["para_dist"], para_msas.shape[1]) for s in batch])
            else:
                out["host_dist"] = torch.stack([jukes_cantor_dist(m) for m in host_msas])
                out["para_dist"] = torch.stack([jukes_cantor_dist(m) for m in para_msas])

        return out

    return collate_fn


def compute_predictions(model, data_loader, device):
    preds_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False):
            if batch is None:
                continue

            batch["host_msa"] = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            batch["sim_time"] = batch["sim_time"].to(device, non_blocking=True)

            if "host_dist" in batch:
                batch["host_dist"] = batch["host_dist"].to(device, non_blocking=True)
                batch["para_dist"] = batch["para_dist"].to(device, non_blocking=True)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
                host_dist=batch.get("host_dist"),
                para_dist=batch.get("para_dist"),
            )
            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if not preds_list:
        return None, None

    return torch.cat(preds_list), torch.cat(labels_list)


def plot_scatter(train_preds, train_labels, val_preds, val_labels, output_dir):
    scatter_dir = output_dir / "scatter_plots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(EVENT_NAMES):
        fig, ax = plt.subplots(figsize=(8, 8))
        tl = train_labels[:, i].numpy()
        tp = train_preds[:, i].numpy()
        vl = val_labels[:, i].numpy()
        vp = val_preds[:, i].numpy()

        ax.scatter(tl, tp, alpha=0.4, s=10, color="blue", label="Train")
        ax.scatter(vl, vp, alpha=0.4, s=10, color="orange", label="Validation")

        lo = min(tl.min(), tp.min(), vl.min(), vp.min())
        hi = max(tl.max(), tp.max(), vl.max(), vp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect prediction")

        ax.set_xlabel("True Labels")
        ax.set_ylabel("Predictions")
        ax.set_title(f"Train vs Val - {name}")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)

        fig.tight_layout()
        out_path = scatter_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_density(train_preds, train_labels, val_preds, val_labels, output_dir, bins=40):
    density_dir = output_dir / "density_plots"
    density_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(EVENT_NAMES):
        fig, ax = plt.subplots(figsize=(10, 6))

        tl = train_labels[:, i].numpy()
        tp = train_preds[:, i].numpy()
        vl = val_labels[:, i].numpy()
        vp = val_preds[:, i].numpy()

        lo = min(tl.min(), tp.min(), vl.min(), vp.min())
        hi = max(tl.max(), tp.max(), vl.max(), vp.max())
        if math.isclose(lo, hi):
            pad = 1e-6 if lo == 0 else abs(lo) * 0.01
            lo -= pad
            hi += pad

        hist_range = (lo, hi)

        ax.hist(
            tl,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:blue",
            label="Train true",
        )
        ax.hist(
            tp,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:cyan",
            label="Train pred",
        )
        ax.hist(
            vl,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:orange",
            label="Val true",
        )
        ax.hist(
            vp,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:red",
            label="Val pred",
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"Final value density - {name}")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

        fig.tight_layout()
        out_path = density_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def write_prediction_csv(preds, labels, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"true_{n}" for n in EVENT_NAMES] + [f"pred_{n}" for n in EVENT_NAMES]
        writer.writerow(header)
        for true_row, pred_row in zip(labels.tolist(), preds.tolist()):
            writer.writerow(true_row + pred_row)


def count_module_indices(state_dict, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    found = set()
    for key in state_dict:
        match = pattern.match(key)
        if match:
            found.add(int(match.group(1)))
    return (max(found) + 1) if found else 0


def infer_model_config(state_dict, hparams):
    hidden_dim = int(hparams.get("hidden_dim", state_dict["host_encoder.embedder.proj.weight"].shape[0]))
    pair_dim = int(hparams.get("pair_dim", state_dict["host_encoder.pair_embedder.proj.weight"].shape[0]))
    cls_dim = int(hparams.get("cls_dim", state_dict["pair_proj_h.weight"].shape[0]))

    axial_layers = int(hparams.get("axial_layers", count_module_indices(state_dict, "host_encoder.evopf_blocks")))
    num_cross_layers = int(
        hparams.get(
            "num_cross_layers",
            hparams.get("cross_layers", count_module_indices(state_dict, "cross_attn_h2p")),
        )
    )

    use_opm = bool(hparams.get("use_opm", False))
    use_dist_matrix = bool(
        hparams.get(
            "use_dist_matrix",
            any(k.startswith("host_encoder.dist_proj.") for k in state_dict),
        )
    )
    use_flexattention = bool(hparams.get("use_flexattention", hparams.get("use_flex", False)))

    return {
        "hidden_dim": hidden_dim,
        "pair_dim": pair_dim,
        "axial_layers": axial_layers,
        "num_cross_layers": num_cross_layers,
        "cls_dim": cls_dim,
        "use_opm": use_opm,
        "use_dist_matrix": use_dist_matrix,
        "use_flexattention": use_flexattention,
    }


def resolve_latest_checkpoint(ckpt_root):
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {ckpt_root}")

    candidates = []
    name_priority = {
        "latest.ckpt": 3,
        "best_val_loss.ckpt": 2,
        "last_epoch.ckpt": 1,
    }

    for ckpt_path in ckpt_root.rglob("*.ckpt"):
        if not ckpt_path.is_file():
            continue
        priority = name_priority.get(ckpt_path.name, 0)
        mtime = ckpt_path.stat().st_mtime
        candidates.append((mtime, priority, ckpt_path))

    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_root}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def build_output_dir(output_root, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    stamp = datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    parent_name = checkpoint_path.parent.name
    run_name = f"{parent_name}_{checkpoint_path.stem}_{stamp}"

    output_root = Path(output_root)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser("CPU post-processing for Co-Phyloformer checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint file. If omitted, latest is auto-detected.")
    parser.add_argument("--ckpt-root", default="checkpoints", help="Root folder scanned for latest checkpoint when --checkpoint is omitted.")
    parser.add_argument(
        "--train-dir",
        default="/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/train_preencoded",
    )
    parser.add_argument(
        "--val-dir",
        default="/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/val_preencoded",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--host-max-leaves", type=int, default=50)
    parser.add_argument("--para-max-leaves", type=int, default=128)
    parser.add_argument("--train-scatter-max", type=int, default=2000)
    parser.add_argument("--output-root", default="post_process_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = resolve_latest_checkpoint(args.ckpt_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[PostProcess] Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        hparams = checkpoint.get("hparams", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        hparams = {}
    else:
        raise ValueError("Unsupported checkpoint format")

    model_cfg = infer_model_config(state_dict, hparams)
    model = Cophyloformer(
        hidden_dim=model_cfg["hidden_dim"],
        pair_dim=model_cfg["pair_dim"],
        cls_dim=model_cfg["cls_dim"],
        axial_layers=model_cfg["axial_layers"],
        use_opm=model_cfg["use_opm"],
        use_dist_matrix=model_cfg["use_dist_matrix"],
        gradient_checkpointing=False,
        num_cross_layers=model_cfg["num_cross_layers"],
        use_flexattention=model_cfg["use_flexattention"],
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    collate_fn = make_collate_fn(
        host_max_leaves=args.host_max_leaves,
        para_max_leaves=args.para_max_leaves,
        use_dist_matrix=model_cfg["use_dist_matrix"],
    )

    train_dataset = LazyCophyloformerDataset(args.train_dir)
    val_dataset = LazyCophyloformerDataset(args.val_dir)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No train .pt files found in: {args.train_dir}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No val .pt files found in: {args.val_dir}")

    output_dir = build_output_dir(args.output_root, checkpoint_path)
    print(f"[PostProcess] Writing outputs to: {output_dir}")

    scatter_n = min(args.train_scatter_max, len(train_dataset))
    scatter_idx = random.sample(range(len(train_dataset)), scatter_n)
    scatter_train_dataset = LazyCophyloformerDataset(
        args.train_dir,
        pt_files=[train_dataset.pt_files[i] for i in scatter_idx],
    )

    train_loader = DataLoader(
        scatter_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print("[PostProcess] Computing train predictions (subset for scatter)...")
    train_preds, train_labels = compute_predictions(model, train_loader, device)
    print("[PostProcess] Computing val predictions (full set)...")
    val_preds, val_labels = compute_predictions(model, val_loader, device)

    if train_preds is None or val_preds is None:
        raise RuntimeError("Could not compute predictions; all batches were empty/invalid")

    plot_scatter(train_preds, train_labels, val_preds, val_labels, output_dir)
    plot_density(train_preds, train_labels, val_preds, val_labels, output_dir)

    train_csv = output_dir / "train_predictions.csv"
    val_csv = output_dir / "val_predictions.csv"
    write_prediction_csv(train_preds, train_labels, train_csv)
    write_prediction_csv(val_preds, val_labels, val_csv)

    train_mae = (train_preds - train_labels).abs().mean(dim=0).tolist()
    val_mae = (val_preds - val_labels).abs().mean(dim=0).tolist()

    metrics = {
        "checkpoint": str(checkpoint_path),
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "output_dir": str(output_dir),
        "n_train_scatter": scatter_n,
        "n_val": int(val_labels.shape[0]),
        "event_names": EVENT_NAMES,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "model_config": model_cfg,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[PostProcess] Saved train predictions: {train_csv}")
    print(f"[PostProcess] Saved val predictions: {val_csv}")
    print(f"[PostProcess] Saved metrics: {metrics_path}")
    print(f"[PostProcess] Saved scatter plots: {output_dir / 'scatter_plots'}")
    print(f"[PostProcess] Saved density plots: {output_dir / 'density_plots'}")


if __name__ == "__main__":
    main()
