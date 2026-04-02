import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb
import glob
import math
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Cophyloformer
from validation import run_full_validation, compute_val_predictions
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric.strategies import DDPStrategy
from sklearn.model_selection import train_test_split

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

seed_everything(42)

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]

# --- env flags ---
use_opm         = os.environ.get("USE_OPM", "0").strip() == "1"
use_dist_matrix = os.environ.get("USE_DIST_MATRIX", "0").strip() == "1"
axial_layers     = int(os.environ.get("AXIAL_LAYERS", "2"))
cross_layers     = int(os.environ.get("CROSS_LAYERS", "1"))
grad_ckpt        = os.environ.get("GRADIENT_CHECKPOINTING", "0").strip() == "1"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID, PAD_ID = 21, 22

def unwrap_model(model):
    """Safely unwrap Fabric and DDP wrappers to get the base PyTorch model."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m

def encode_sequence(sequence, max_len=256):
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
            print(f"[Warning] sample {self.pt_files[idx]} missing host or parasite MSAs, skipping")
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
            "host_msa":    torch.stack([encode_sequence(s) for s in sample["host_msas"].values()]),
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


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        print("[Warning] all samples in batch were invalid, returning None")
        return None

    def pad(msas, pad_val=PAD_ID):
        max_n = max(m.shape[0] for m in msas)
        return torch.stack([F.pad(m, (0, 0, 0, max_n - m.shape[0]), value=pad_val) for m in msas])

    host_msas = pad([s["host_msa"] for s in batch])
    para_msas = pad([s["parasite_msa"] for s in batch])
    out = {
        "host_msa":     host_msas,
        "parasite_msa": para_msas,
        "labels":       torch.stack([s["labels"] for s in batch]),
        "mappings":     [s["mappings"] for s in batch],
        "sim_time":     torch.stack([s["sim_time"] for s in batch]),
    }
    if use_dist_matrix:
        if "host_dist" in batch[0]:
            def pad_dist(d, n):
                if d.shape[0] == n:
                    return d
                out = torch.zeros(n, n, dtype=d.dtype)
                out[:d.shape[0], :d.shape[0]] = d
                return out
            out["host_dist"] = torch.stack([pad_dist(s["host_dist"], host_msas.shape[1]) for s in batch])
            out["para_dist"] = torch.stack([pad_dist(s["para_dist"], para_msas.shape[1]) for s in batch])
        else:
            out["host_dist"] = torch.stack([jukes_cantor_dist(m) for m in host_msas])
            out["para_dist"] = torch.stack([jukes_cantor_dist(m) for m in para_msas])
    return out


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(fabric, model, optimizer, scheduler, epoch, global_step, epoch_step,
                    val_loss, hparams, ckpt_dir, best_val, end_of_epoch=False, best_only=False):
    """Save a checkpoint following the Phyloformer-2 schema.

    Unless best_only=True, always writes latest.ckpt (and last_epoch.ckpt at end of epoch).
    When val_loss improves, always writes best_val_loss.ckpt.
    Returns the (possibly updated) best_val.
    """
    new_best_val = best_val
    if fabric.is_global_zero:
        # Skip building state entirely if best_only and no improvement
        if best_only and val_loss >= best_val:
            pass
        else:
            state = {
                "model":      unwrap_model(model).state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "step":       global_step,
                "epoch_step": 0 if end_of_epoch else epoch_step,
                "val_loss":   val_loss,
                "hparams":    hparams,
            }
            os.makedirs(ckpt_dir, exist_ok=True)
            if not best_only:
                torch.save(state, os.path.join(ckpt_dir, "latest.ckpt"))
                if end_of_epoch:
                    torch.save(state, os.path.join(ckpt_dir, "last_epoch.ckpt"))
            if val_loss < best_val:
                torch.save(state, os.path.join(ckpt_dir, "best_val_loss.ckpt"))
                new_best_val = val_loss
                print(f"  [Checkpoint] new best val={new_best_val:.6f} → best_val_loss.ckpt")
    fabric.barrier()
    return new_best_val


def load_checkpoint(fabric, model, optimizer, scheduler, ckpt):
    """Restore model, optimizer and scheduler from a checkpoint dict.

    Returns (start_epoch, global_step, epoch_step, best_val).
    """
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"]
    global_step = ckpt["step"] + 1
    epoch_step  = ckpt.get("epoch_step", 0)
    best_val    = ckpt["val_loss"]
    if fabric.is_global_zero:
        print(f"[Resume] epoch={start_epoch}  step={global_step}  best_val={best_val:.6f}")
    return start_epoch, global_step, epoch_step, best_val


def plot_scatter(train_preds, train_labels, val_preds, val_labels, event_names,
                 save_dir="scatter_plots"):
    """Combined train+val scatter plots per event, saved to disk and returned as wandb Images."""
    os.makedirs(save_dir, exist_ok=True)
    images = {}
    for i, name in enumerate(event_names):
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
        ax.set_title(f"Train vs Val — {name}")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        png_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(png_path, dpi=150)
        print(f"  [Scatter] saved {png_path}")
        images[f"scatter/{name}"] = wandb.Image(fig)
        plt.close(fig)
    return images


def main(fabric: Fabric, ckpt_to_load=None):
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    preencoded_dir = "/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/test/"
    epochs         = 500
    batch_size     = 8
    grad_accum     = 32
    lr             = 2e-4
    wd             = 0.05
    huber_delta    = 1.0
    under_penalty  = 1.5
    tail_weight    = 1.0
    mid_epoch_vals = int(os.environ.get("MID_EPOCH_VALS", "8"))
    num_workers    = int(os.environ.get("NUM_WORKERS", "8"))
    ckpt_dir       = os.environ.get("CKPT_DIR", "checkpoints")
    device         = fabric.device

    hparams = {
        "preencoded_dir":  preencoded_dir,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "grad_accum":      grad_accum,
        "lr":              lr,
        "wd":              wd,
        "huber_delta":     huber_delta,
        "under_penalty":   under_penalty,
        "tail_weight":     tail_weight,
        "axial_layers":    axial_layers,
        "use_opm":         use_opm,
        "use_dist_matrix": use_dist_matrix,
        "pair_dim":        32,
        "mid_epoch_vals":  mid_epoch_vals,
        "ckpt_dir":        ckpt_dir,
    }

    event_loss_weights = torch.tensor([1.0, 1.5, 1.0, 1.0], device=device)

    def asymmetric_huber(pred, target):
        err = target - pred
        abs_err = err.abs()
        loss = torch.where(abs_err < huber_delta,
                           0.5 * abs_err ** 2,
                           huber_delta * (abs_err - 0.5 * huber_delta))
        weight = torch.where(err > 0, torch.full_like(err, under_penalty), torch.ones_like(err))
        return loss * weight

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    dataset = LazyCophyloformerDataset(preencoded_dir)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              persistent_workers=True, prefetch_factor=4, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers,
                              persistent_workers=False,
                              prefetch_factor=2 if num_workers > 0 else None, pin_memory=True)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    train_sampler = train_loader.sampler

    # -------------------------------------------------------------------------
    # Model + optimizer + scheduler
    # -------------------------------------------------------------------------
    model = Cophyloformer(
        pair_dim=32, axial_layers=axial_layers,
        use_opm=use_opm, use_dist_matrix=use_dist_matrix,
        gradient_checkpointing=grad_ckpt,
        num_cross_layers=cross_layers,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    opt_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps         = epochs * opt_steps_per_epoch
    warmup_steps    = int(0.15 * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1)

    # -------------------------------------------------------------------------
    # Resume from checkpoint if provided
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val    = float("inf")

    if ckpt_to_load is not None:
        start_epoch, global_step, _, best_val = load_checkpoint(
            fabric, model, optimizer, scheduler, ckpt_to_load
        )

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    if fabric.is_global_zero:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "cophylo_team"),
            project=os.environ.get("WANDB_PROJECT", "CoPhyloformer"),
            name=os.environ.get("WANDB_NAME", "run"),
            group=os.environ.get("WANDB_NAME", "run"),
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={
                **hparams,
                "total_steps":  total_steps,
                "warmup_steps": warmup_steps,
                "dataset_size": len(dataset),
                "start_epoch":  start_epoch,
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def run_mid_val(label, epoch, global_step, epoch_step, best_val):
        """Run validation, log to wandb, save best_val_loss.ckpt only if improved."""
        fabric.barrier()
        vr = run_full_validation(fabric, model, val_loader, asymmetric_huber,
                                  EVENT_NAMES, device,
                                  event_loss_weights=event_loss_weights,
                                  tail_weight_scale=tail_weight)
        fabric.barrier()
        if fabric.is_global_zero:
            mae_s = " | ".join(f"{EVENT_NAMES[i]}: {vr['val_mae'][i]:.4f}" for i in range(len(EVENT_NAMES)))
            print(f"  [{label}] loss={vr['val_loss']:.6f}  MAE → {mae_s}  lr={optimizer.param_groups[0]['lr']:.2e}")
            wandb.log({"val/loss_mid": vr["val_loss"],
                       **{f"val/MAE_mid/{EVENT_NAMES[i]}": vr["val_mae"][i] for i in range(len(EVENT_NAMES))}},
                      step=global_step)
        return save_checkpoint(
            fabric, model, optimizer, scheduler,
            epoch, global_step, epoch_step, vr["val_loss"],
            hparams, ckpt_dir, best_val, end_of_epoch=False, best_only=True,
        )

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)

        # evenly-spaced mid-epoch validation trigger steps
        mid_val_steps = {
            math.ceil(k * opt_steps_per_epoch / (mid_epoch_vals + 1))
            for k in range(1, mid_epoch_vals + 1)
        }

        if fabric.is_global_zero:
            print(f"\nEpoch {epoch+1}/{epochs}")

        # beginning-of-epoch validation (no checkpoint unless new best)
        best_val = run_mid_val("Begin", epoch, global_step, 0, best_val)

        total_loss  = 0.0
        num_batches = 0
        epoch_step  = 0

        sum_abs   = torch.zeros(len(EVENT_NAMES), device=device)
        n_samples = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                     desc=f"Epoch {epoch+1}", leave=False):
            if batch is None:
                continue
            batch["host_msa"]     = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["sim_time"]     = batch["sim_time"].to(device, non_blocking=True)
            batch["labels"]       = batch["labels"].to(device, non_blocking=True)
            if use_dist_matrix:
                batch["host_dist"] = batch["host_dist"].to(device, non_blocking=True)
                batch["para_dist"] = batch["para_dist"].to(device, non_blocking=True)

            if batch_idx % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch["host_msa"], batch["parasite_msa"], batch["mappings"],
                batch["sim_time"],
                host_dist=batch.get("host_dist"), para_dist=batch.get("para_dist"),
            )

            tw = 1.0 + tail_weight * batch["labels"]
            loss = sum(
                event_loss_weights[i] * (asymmetric_huber(outputs[:, i], batch["labels"][:, i]) * tw[:, i]).mean()
                for i in range(len(EVENT_NAMES))
            )
            fabric.backward(loss / grad_accum)

            is_accum = ((batch_idx + 1) % grad_accum == 0)
            is_last  = ((batch_idx + 1) == len(train_loader))
            if is_accum or is_last:
                gnorm = fabric.clip_gradients(model, optimizer, max_norm=0.5, error_if_nonfinite=False)
                if torch.isfinite(gnorm):
                    optimizer.step()
                    scheduler.step()
                    epoch_step  += 1
                    global_step += 1

                    # mid-epoch validation (no checkpoint unless new best)
                    if epoch_step in mid_val_steps:
                        mid_val_steps.discard(epoch_step)
                        pct = int(round(epoch_step / opt_steps_per_epoch * 100))
                        best_val = run_mid_val(f"Val {pct:3d}%", epoch, global_step, epoch_step, best_val)
                else:
                    if fabric.is_global_zero:
                        print(f"[Warning] non-finite grad norm {gnorm:.2e} at batch {batch_idx+1}, skipping")
                    optimizer.zero_grad(set_to_none=True)

            total_loss  += loss.item()
            num_batches += 1
            with torch.no_grad():
                sum_abs   += (outputs.detach() - batch["labels"]).abs().sum(dim=0)
                n_samples += outputs.shape[0]

        # --- End of epoch ---
        epoch_loss = fabric.all_reduce(torch.tensor(total_loss / max(1, num_batches), device=device),
                                       reduce_op="mean").item()
        global_abs = fabric.all_reduce(sum_abs, reduce_op="sum")
        global_n   = int(fabric.all_reduce(torch.tensor(n_samples, device=device), reduce_op="sum").item())
        mae = (global_abs / max(1, global_n)).cpu().tolist()

        if fabric.is_global_zero:
            print(f"  train loss={epoch_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {mae[i]:.4f}" for i in range(len(EVENT_NAMES))))

        # end-of-epoch validation + checkpoint
        fabric.barrier()
        vr = run_full_validation(fabric, model, val_loader, asymmetric_huber,
                                  EVENT_NAMES, device,
                                  event_loss_weights=event_loss_weights,
                                  tail_weight_scale=tail_weight)
        fabric.barrier()
        val_loss = vr["val_loss"]
        val_mae  = vr["val_mae"]

        if fabric.is_global_zero:
            print(f"  val  loss={val_loss:.6f}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {val_mae[i]:.4f}" for i in range(len(EVENT_NAMES))))
            ep_step = (epoch + 1) * opt_steps_per_epoch
            wandb.log({
                "epoch": epoch + 1, "train/loss": epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/MAE/{EVENT_NAMES[i]}": mae[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step)
            wandb.log({
                "epoch": epoch + 1, "val/loss": val_loss,
                **{f"val/MAE/{EVENT_NAMES[i]}": val_mae[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step + 1)

        best_val = save_checkpoint(
            fabric, model, optimizer, scheduler,
            epoch, global_step, epoch_step, val_loss,
            hparams, ckpt_dir, best_val, end_of_epoch=True,
        )

    # -------------------------------------------------------------------------
    # Final save + scatter plots
    # -------------------------------------------------------------------------
    if fabric.is_global_zero:
        torch.save(unwrap_model(model).state_dict(), "cophyloformer_final.pth")

        raw_model = unwrap_model(model)
        raw_model.eval()

        # val scatter — all val data
        scatter_val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        val_preds, val_labels = compute_val_predictions(raw_model, scatter_val_loader, device)

        # train scatter — up to 2000 random samples
        scatter_n = min(2000, len(train_idx))
        scatter_idx = random.sample(train_idx, scatter_n)
        scatter_train_ds = LazyCophyloformerDataset(
            preencoded_dir,
            pt_files=[dataset.pt_files[i] for i in scatter_idx],
        )
        scatter_train_loader = DataLoader(
            scatter_train_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        train_preds, train_labels = compute_val_predictions(raw_model, scatter_train_loader, device)

        if val_preds is not None and train_preds is not None:
            wandb.log(plot_scatter(train_preds, train_labels, val_preds, val_labels, EVENT_NAMES))

        if val_preds is not None:
            import csv
            csv_path = "val_predictions.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = (
                    [f"true_{n}" for n in EVENT_NAMES]
                    + [f"pred_{n}" for n in EVENT_NAMES]
                )
                writer.writerow(header)
                for true_row, pred_row in zip(val_labels.tolist(), val_preds.tolist()):
                    writer.writerow(true_row + pred_row)
            print(f"  [CSV] saved {csv_path}  ({len(val_labels)} rows)")
            wandb.save(csv_path)

        wandb.finish()
        print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train Co-Phyloformer")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", description="Train from scratch")
    resumer = subparsers.add_parser("resume", description="Resume from a checkpoint")
    resumer.add_argument("checkpoint", help="Path to .ckpt file (latest.ckpt or last_epoch.ckpt)")
    args = parser.parse_args()

    ckpt_to_load = None
    if args.command == "resume":
        ckpt_to_load = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # If resuming from an end-of-epoch checkpoint, start from the next epoch
        if "last_epoch.ckpt" in args.checkpoint:
            ckpt_to_load["epoch"] += 1

    def _main(fabric: Fabric):
        main(fabric, ckpt_to_load=ckpt_to_load)

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    fabric.launch(_main)
