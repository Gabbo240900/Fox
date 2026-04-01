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
axial_layers    = int(os.environ.get("AXIAL_LAYERS", "2"))

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID, PAD_ID = 21, 22


def encode_sequence(sequence, max_len=500):
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
            all_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
            valid = []
            for p in all_files:
                try:
                    s = torch.load(p, map_location="cpu", weights_only=False)
                    if s.get("host_msas") and s.get("parasite_msas"):
                        valid.append(p)
                except Exception:
                    continue
            self.pt_files = valid
        else:
            self.pt_files = list(pt_files)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx], map_location="cpu", weights_only=False)

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


def save_checkpoint(fabric, model, optimizer, scheduler, epoch, val_loss, path, history=None, batch_idx=None):
    state = {
        "model": model, "optimizer": optimizer,
        "lr_scheduler": scheduler.state_dict(),
        "epoch": epoch, "val_loss": val_loss,
    }
    if batch_idx is not None:
        state["batch_idx"] = batch_idx
    if history is not None:
        state["history"] = history
    fabric.save(path, state)


def load_checkpoint(fabric, model, optimizer, scheduler, path):
    state = {"model": model, "optimizer": optimizer}
    meta = fabric.load(path, state)
    if scheduler is not None and "lr_scheduler" in meta:
        scheduler.load_state_dict(meta["lr_scheduler"])
    return (
        meta.get("epoch", 0),
        meta.get("val_loss", float("inf")),
        meta.get("batch_idx", None),
        meta.get("history", None),
    )


def plot_scatter(preds, labels, split, event_names):
    images = {}
    for i, name in enumerate(event_names):
        fig, ax = plt.subplots(figsize=(4, 4))
        x = labels[:, i].numpy()
        y = preds[:, i].numpy()
        ax.scatter(x, y, alpha=0.3, s=4)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(f"{split} — {name}")
        fig.tight_layout()
        images[f"scatter/{split}/{name}"] = wandb.Image(fig)
        plt.close(fig)
    return images


def main(fabric: Fabric):
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    preencoded_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/test/"
    epochs          = 500
    batch_size      = 32
    grad_accum      = 8
    lr              = 2e-4
    wd              = 0.05
    huber_delta     = 1.0
    under_penalty   = 1.5
    tail_weight     = 1.0
    mid_epoch_vals  = max(0, int(os.environ.get("MID_EPOCH_VALS", "2")))
    num_workers     = int(os.environ.get("NUM_WORKERS", "8"))
    disable_ckpt    = os.environ.get("DISABLE_CHECKPOINTS", "0").strip() == "1"
    gradient_ckpt   = os.environ.get("GRADIENT_CHECKPOINTING", "0").strip() == "1"
    checkpoint_dir  = "checkpoints"
    device          = fabric.device

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
        gradient_checkpointing=gradient_ckpt,
        use_opm=use_opm, use_dist_matrix=use_dist_matrix,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = int(0.15 * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1)

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    start_epoch = 0
    start_batch = 0
    best_val    = float("inf")
    history     = {"epoch_losses": [], "val_loss_history": [],
                   "mae_history": [], "val_mae_history": []}

    resume_path = os.environ.get("RESUME_CKPT")
    if resume_path and os.path.exists(resume_path):
        if fabric.is_global_zero:
            print(f"[Resume] Loading {resume_path}")
        loaded_epoch, loaded_val, loaded_batch, loaded_hist = load_checkpoint(
            fabric, model, optimizer, scheduler, resume_path)
        start_epoch = loaded_epoch
        start_batch = (loaded_batch + 1) if loaded_batch is not None else 0
        best_val    = loaded_val
        if loaded_hist:
            history = loaded_hist
        if fabric.is_global_zero:
            print(f"[Resume] epoch={start_epoch}  best_val={best_val:.6f}  batch={loaded_batch}")

    # -------------------------------------------------------------------------
    # WandB — always fresh run, replay history so every offline file is complete
    # -------------------------------------------------------------------------
    if fabric.is_global_zero:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "cophylo_team"),
            project=os.environ.get("WANDB_PROJECT", "CoPhyloformer"),
            name=os.environ.get("WANDB_NAME", "run"),
            group=os.environ.get("WANDB_NAME", "run"),
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={
                "epochs": epochs, "batch_size": batch_size, "lr": lr, "wd": wd,
                "grad_accum": grad_accum, "total_steps": total_steps,
                "warmup_steps": warmup_steps, "axial_layers": axial_layers,
                "use_opm": use_opm, "use_dist_matrix": use_dist_matrix,
                "pair_dim": 32, "dataset_size": len(dataset), "start_epoch": start_epoch,
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )
        # Replay history so this offline file is self-contained
        for ep_i, ep_loss in enumerate(history["epoch_losses"]):
            ep_step = (ep_i + 1) * steps_per_epoch
            log = {"epoch": ep_i + 1, "train/loss": ep_loss}
            if ep_i < len(history["mae_history"]):
                for j, n in enumerate(EVENT_NAMES):
                    log[f"train/MAE/{n}"] = history["mae_history"][ep_i][j]
            wandb.log(log, step=ep_step)
            if ep_i < len(history["val_loss_history"]):
                vlog = {"epoch": ep_i + 1, "val/loss": history["val_loss_history"][ep_i]}
                if ep_i < len(history["val_mae_history"]):
                    for j, n in enumerate(EVENT_NAMES):
                        vlog[f"val/MAE/{n}"] = history["val_mae_history"][ep_i][j]
                wandb.log(vlog, step=ep_step + 1)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        opt_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)

        # mid-epoch val trigger steps
        mid_val_steps = {
            math.ceil(k * opt_steps_per_epoch / (mid_epoch_vals + 1))
            for k in range(1, mid_epoch_vals + 1)
        }
        if fabric.is_global_zero:
            print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()
        total_loss   = 0.0
        num_batches  = 0
        opt_step     = (start_batch // grad_accum) if (epoch == start_epoch and start_batch > 0) else 0
        mid_val_steps = {s for s in mid_val_steps if s > opt_step}

        sum_abs = torch.zeros(len(EVENT_NAMES), device=device)
        n_samples = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                     desc=f"Epoch {epoch+1}", leave=False):
            # skip already-processed batches on resume
            if epoch == start_epoch and batch_idx < start_batch:
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
                    opt_step += 1
                else:
                    if fabric.is_global_zero:
                        print(f"[Warning] non-finite grad norm {gnorm:.2e} at batch {batch_idx+1}, skipping")
                    optimizer.zero_grad(set_to_none=True)

            total_loss  += loss.item()
            num_batches += 1
            with torch.no_grad():
                sum_abs  += (outputs.detach() - batch["labels"]).abs().sum(dim=0)
                n_samples += outputs.shape[0]

            # mid-epoch validation
            if opt_step in mid_val_steps:
                mid_val_steps.discard(opt_step)
                fabric.barrier()
                vr = run_full_validation(fabric, model, val_loader, asymmetric_huber,
                                         EVENT_NAMES, device,
                                         event_loss_weights=event_loss_weights,
                                         tail_weight_scale=tail_weight)
                fabric.barrier()
                if fabric.is_global_zero:
                    pct = int(round(opt_step / opt_steps_per_epoch * 100))
                    mae_s = " | ".join(f"{EVENT_NAMES[i]}: {vr['val_mae'][i]:.4f}" for i in range(len(EVENT_NAMES)))
                    print(f"  [Val {pct:3d}%] loss={vr['val_loss']:.6f}  MAE → {mae_s}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")
                    gstep = epoch * opt_steps_per_epoch + opt_step
                    wandb.log({"val/loss_mid": vr["val_loss"],
                               **{f"val/MAE_mid/{EVENT_NAMES[i]}": vr["val_mae"][i] for i in range(len(EVENT_NAMES))}},
                              step=gstep)

        # --- End of epoch ---
        epoch_loss = fabric.all_reduce(torch.tensor(total_loss / max(1, num_batches), device=device),
                                       reduce_op="mean").item()
        global_abs = fabric.all_reduce(sum_abs, reduce_op="sum")
        global_n   = int(fabric.all_reduce(torch.tensor(n_samples, device=device), reduce_op="sum").item())
        mae = (global_abs / max(1, global_n)).cpu().tolist()

        history["epoch_losses"].append(epoch_loss)
        history["mae_history"].append(mae)

        if fabric.is_global_zero:
            print(f"  train loss={epoch_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {mae[i]:.4f}" for i in range(len(EVENT_NAMES))))

        # end-of-epoch validation
        fabric.barrier()
        vr = run_full_validation(fabric, model, val_loader, asymmetric_huber,
                                  EVENT_NAMES, device,
                                  event_loss_weights=event_loss_weights,
                                  tail_weight_scale=tail_weight)
        fabric.barrier()
        val_loss = vr["val_loss"]
        val_mae  = vr["val_mae"]
        history["val_loss_history"].append(val_loss)
        history["val_mae_history"].append(val_mae)

        if fabric.is_global_zero:
            print(f"  val  loss={val_loss:.6f}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {val_mae[i]:.4f}" for i in range(len(EVENT_NAMES))))
            ep_step = (epoch + 1) * steps_per_epoch
            wandb.log({
                "epoch": epoch + 1, "train/loss": epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/MAE/{EVENT_NAMES[i]}": mae[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step)
            wandb.log({
                "epoch": epoch + 1, "val/loss": val_loss,
                **{f"val/MAE/{EVENT_NAMES[i]}": val_mae[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step + 1)

        # checkpoint
        if not disable_ckpt:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_end.pth")
            save_checkpoint(fabric, model, optimizer, scheduler,
                            epoch + 1, val_loss, ckpt_path, history=history)
            if fabric.is_global_zero:
                print(f"  [Checkpoint] saved {ckpt_path}")

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(checkpoint_dir, "best.pth")
                save_checkpoint(fabric, model, optimizer, scheduler,
                                epoch + 1, best_val, best_path, history=history)
                if fabric.is_global_zero:
                    print(f"  [Checkpoint] new best val={best_val:.6f} → {best_path}")
        elif fabric.is_global_zero and val_loss < best_val:
            best_val = val_loss

        model.train()

    # -------------------------------------------------------------------------
    # Final save + scatter plots
    # -------------------------------------------------------------------------
    if fabric.is_global_zero:
        torch.save(model.state_dict(), "cophyloformer_final.pth")

        raw_model = fabric.unwrap_model(model)
        raw_model.eval()

        # val scatter — all val data
        scatter_val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        val_preds, val_labels = compute_val_predictions(raw_model, scatter_val_loader, device)

        # train scatter — up to 2000 random samples, no masking
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

        if val_preds is not None:
            wandb.log(plot_scatter(val_preds, val_labels, "val", EVENT_NAMES))
            # save validation predictions to CSV
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

        if train_preds is not None:
            wandb.log(plot_scatter(train_preds, train_labels, "train", EVENT_NAMES))

        wandb.finish()
        print("Training complete.")


if __name__ == "__main__":
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    fabric.launch(main)
