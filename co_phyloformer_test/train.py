"""
train.py - Multi-GPU DDP training script for Co-Phyloformer
==============================================================
Supports single-GPU, multi-GPU (torchrun), and JeanZay SLURM (srun) runs.

Single GPU:
    python train.py --data_dir /path/to/Datasets

4 GPUs on one node (torchrun):
    torchrun --nproc_per_node=4 train.py --data_dir /path/to/Datasets

JeanZay H100 (submitted via slurm_train.sh, uses srun python train.py):
    sbatch slurm_train.sh
"""

import os
import sys
import argparse
import random
import time
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optional WandB — gracefully disabled if not installed or WANDB_MODE=disabled
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model import CoPhyloformer
    _MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import CoPhyloformer: {e}")
    _MODEL_AVAILABLE = False
    CoPhyloformer = None

try:
    from dataset import get_dataloaders
    _DATASET_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import get_dataloaders: {e}")
    _DATASET_AVAILABLE = False
    get_dataloaders = None


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """Initialise the distributed process group and return (rank, local_rank, world_size).

    Handles two launch patterns:
    - torchrun: sets RANK / LOCAL_RANK / WORLD_SIZE automatically.
    - JeanZay srun (srun python train.py): SLURM sets SLURM_PROCID /
      SLURM_LOCALID / SLURM_NTASKS; MASTER_ADDR / MASTER_PORT must be set
      by the job script before srun (see slurm_train.sh).

    Returns (0, 0, 1) when no distributed environment is detected so that
    single-GPU runs require no code changes.
    """
    # torchrun path
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

    # srun path (JeanZay): read SLURM env vars and inject as standard dist vars
    elif "SLURM_PROCID" in os.environ:
        rank       = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        # dist.init_process_group("env://") needs these set
        os.environ["RANK"]       = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        # Single-process — no DDP
        return 0, 0, 1

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average a scalar tensor across all ranks in-place and return it."""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return tensor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0) -> None:
    """Set deterministic seeds. Each rank gets a unique seed offset."""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Allow cuDNN auto-tuner on H100 for speed (non-deterministic but faster).
    # Set to True / False for fully reproducible (but slower) runs.
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def kl_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence loss between predicted and target probability distributions."""
    log_pred = predictions.clamp(min=1e-9).log()
    return F.kl_div(log_pred, targets, reduction="batchmean")


def mae_per_component(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-component MAE. predictions and targets: [N, 4]."""
    return (predictions - targets).abs().mean(dim=0)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
) -> None:
    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
            "best_val_loss":        best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch    = checkpoint["epoch"] + 1
    best_val_loss  = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Single-epoch loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer=None,
    scaler=None,
    scheduler=None,
    accumulate_grad: int = 8,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    world_size: int = 1,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Run one training or validation epoch.

    Training mode: pass ``optimizer`` (non-None).
    Validation mode: pass ``optimizer=None``.

    Returns:
        (mean_loss, all_predictions [N,4], all_targets [N,4]) on CPU.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples  = 0
    all_preds  = []
    all_tgts   = []
    step       = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    if is_train:
        optimizer.zero_grad()

    no_grad_ctx = nullcontext() if is_train else torch.no_grad()

    # For DistributedSampler: set epoch so each epoch has a different shuffle
    if is_train and world_size > 1 and hasattr(loader.sampler, "set_epoch"):
        # epoch is passed via the loader's sampler — caller sets it externally
        pass

    with no_grad_ctx:
        for batch in loader:
            # collate_fn returns a list of dicts — take the first (batch_size=1)
            sample        = batch[0] if isinstance(batch, list) else batch
            host_seqs     = sample["host_seqs"]
            parasite_seqs = sample["parasite_seqs"]
            mappings      = sample["mappings"]
            labels        = sample["labels"]

            # Move labels to device
            labels = labels.to(device, non_blocking=True)
            if labels.dim() == 2:
                labels = labels.squeeze(0)

            # Move sequence tensors to device
            host_seqs = {
                k: v.to(device, non_blocking=True)
                for k, v in host_seqs.items()
            }
            parasite_seqs = {
                k: v.to(device, non_blocking=True)
                for k, v in parasite_seqs.items()
            }

            # Forward
            with autocast_ctx:
                predictions = model(host_seqs, parasite_seqs, mappings)
                if predictions.dim() == 2:
                    predictions = predictions.squeeze(0)
                loss        = kl_loss(predictions, labels)
                loss_scaled = loss / accumulate_grad

            if is_train:
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if (step + 1) % accumulate_grad == 0:
                    _optimizer_step(optimizer, scaler, model)

            total_loss += loss.item()
            n_samples  += 1
            step       += 1

            all_preds.append(predictions.detach().cpu().float())
            all_tgts.append(labels.detach().cpu().float())

    # Flush remaining gradients at epoch end
    if is_train and (step % accumulate_grad != 0):
        _optimizer_step(optimizer, scaler, model)

    # Average loss across all ranks
    mean_loss = torch.tensor(total_loss / max(n_samples, 1), device=device)
    mean_loss = all_reduce_mean(mean_loss, world_size)

    preds_tensor  = torch.stack(all_preds)  if all_preds else torch.empty(0, 4)
    tgts_tensor   = torch.stack(all_tgts)   if all_tgts  else torch.empty(0, 4)

    return mean_loss.item(), preds_tensor, tgts_tensor


def _optimizer_step(optimizer, scaler, model) -> None:
    """Unscale → clip → step → zero_grad, handling both AMP and non-AMP."""
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Full DDP training procedure for Co-Phyloformer."""
    if not _MODEL_AVAILABLE:
        raise RuntimeError("model.py could not be imported.")
    if not _DATASET_AVAILABLE:
        raise RuntimeError("dataset.py could not be imported.")

    # ---- Distributed setup -----------------------------------------------
    rank, local_rank, world_size = init_distributed()
    main = is_main(rank)

    # ---- Device -------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if main:
        print(f"[INFO] world_size={world_size}  rank={rank}  device={device}")

    # ---- Reproducibility ----------------------------------------------------
    set_seed(args.seed, rank=rank)

    # ---- Mixed precision (BF16 preferred on H100) ---------------------------
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    # GradScaler is needed for FP16 but NOT for BF16
    scaler    = (torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None)
    if main and use_amp:
        print(f"[INFO] AMP enabled — dtype={amp_dtype}")

    # ---- WandB (rank 0 only, honours WANDB_MODE env var) --------------------
    use_wandb = False
    if main and _WANDB_AVAILABLE and os.environ.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            project = os.environ.get("WANDB_PROJECT", "CoPhyloformer"),
            entity  = os.environ.get("WANDB_ENTITY",  None),
            name    = os.environ.get("WANDB_NAME",    None),
            config  = vars(args),
            resume  = "allow",
        )
        use_wandb = True
        print(f"[INFO] WandB run: {wandb.run.name}  (mode={os.environ.get('WANDB_MODE')})")

    # ---- Checkpoints --------------------------------------------------------
    ckpt_dir      = Path(args.checkpoint_dir)
    best_ckpt     = str(ckpt_dir / "best_model.pt")
    last_ckpt     = str(ckpt_dir / "last_model.pt")
    if main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    if main:
        print(f"[INFO] Loading data from: {args.data_dir}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir        = args.data_dir,
        batch_size      = 1,          # variable tree sizes → process one at a time
        train_frac      = args.train_frac,
        val_frac        = args.val_frac,
        seed            = args.seed,
        max_seq_len     = args.max_seq_len,
        num_workers     = args.num_workers,
        pin_memory      = (device.type == "cuda"),
        persistent_workers = (args.num_workers > 0),
        prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None,
        rank            = rank,
        world_size      = world_size,
        manifest_file   = args.manifest_file,
    )

    if main:
        print(
            f"[INFO] Splits — train: {len(train_loader.dataset)}  "
            f"val: {len(val_loader.dataset)}  "
            f"test: {len(test_loader.dataset)}"
        )

    # ---- Model --------------------------------------------------------------
    model = CoPhyloformer(
        d_model               = args.d_model,
        nhead                 = args.nhead,
        num_layers            = args.num_layers,
        dropout               = args.dropout,
        gradient_checkpointing = args.gradient_checkpointing,
    ).to(device)

    # torch.compile for H100 (requires PyTorch >= 2.0)
    if args.compile and hasattr(torch, "compile"):
        if main:
            print("[INFO] Compiling model with torch.compile …")
        model = torch.compile(model)

    # Wrap in DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids           = [local_rank],
            output_device        = local_rank,
            find_unused_parameters = False,
        )

    if main:
        raw = model.module if hasattr(model, "module") else model
        n   = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"[INFO] Trainable parameters: {n:,}")

    # ---- Optimiser & scheduler ----------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),   # slightly larger beta2, common for LLM-like models
        fused        = (device.type == "cuda"),  # fused AdamW is faster on CUDA
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # ---- Resume -------------------------------------------------------------
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        if not os.path.isfile(args.resume):
            if main:
                print(f"[WARNING] Resume checkpoint not found: {args.resume}")
        else:
            # Load on main first, then broadcast weights so all ranks are in sync
            if main:
                start_epoch, best_val_loss = load_checkpoint(
                    args.resume, model, optimizer, scheduler, scaler, device
                )
                print(f"[INFO] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
            if world_size > 1:
                # Broadcast scalar state from rank 0
                info = torch.tensor([start_epoch, best_val_loss], device=device)
                dist.broadcast(info, src=0)
                start_epoch   = int(info[0].item())
                best_val_loss = info[1].item()
                # Broadcast model parameters
                for p in (model.module if hasattr(model, "module") else model).parameters():
                    dist.broadcast(p.data, src=0)

    # ---- Training loop ------------------------------------------------------
    EVENTS = ["Speciation", "HGT", "Loss", "Duplication"]
    patience              = args.patience
    epochs_no_improve     = 0

    if main:
        print("\n" + "=" * 90)
        print(f"{'Ep':>5}  {'Train KL':>10}  {'Val KL':>10}  "
              f"{'Spec MAE':>10}  {'HGT MAE':>9}  "
              f"{'Loss MAE':>9}  {'Dup MAE':>9}  {'LR':>9}  {'Time':>7}")
        print("=" * 90)

    for epoch in range(start_epoch, args.epochs):
        # Set epoch on DistributedSampler so shuffle is different each epoch
        if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()

        # ---- Train ----------------------------------------------------------
        train_loss, _, _ = run_epoch(
            model          = model,
            loader         = train_loader,
            device         = device,
            optimizer      = optimizer,
            scaler         = scaler,
            scheduler      = scheduler,
            accumulate_grad = args.accumulate_grad,
            amp_dtype      = amp_dtype,
            use_amp        = use_amp,
            world_size     = world_size,
        )

        # ---- Validation -----------------------------------------------------
        val_loss, val_preds, val_tgts = run_epoch(
            model          = model,
            loader         = val_loader,
            device         = device,
            optimizer      = None,
            scaler         = None,
            accumulate_grad = 1,
            amp_dtype      = amp_dtype,
            use_amp        = use_amp,
            world_size     = world_size,
        )

        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        if main:
            if val_preds.shape[0] > 0:
                mae = mae_per_component(val_preds, val_tgts)
                mae_str = "  ".join(f"{mae[i].item():.4f}" for i in range(4))
            else:
                mae_str = "  N/A"
                mae     = torch.zeros(4)

            print(
                f"{epoch+1:>5}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                f"{mae[0].item():>10.4f}  {mae[1].item():>9.4f}  "
                f"{mae[2].item():>9.4f}  {mae[3].item():>9.4f}  "
                f"{lr:>9.2e}  {elapsed:>6.1f}s"
            )

            # WandB logging
            if use_wandb:
                log_dict = {
                    "epoch":       epoch + 1,
                    "train/kl_loss": train_loss,
                    "val/kl_loss":   val_loss,
                    "lr":            lr,
                    "val/mae_speciation":  mae[0].item(),
                    "val/mae_hgt":         mae[1].item(),
                    "val/mae_loss":        mae[2].item(),
                    "val/mae_duplication": mae[3].item(),
                    "val/mae_mean":        mae.mean().item(),
                }
                wandb.log(log_dict, step=epoch + 1)

            # Checkpoint
            save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler, epoch, best_val_loss)

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                epochs_no_improve = 0
                save_checkpoint(best_ckpt, model, optimizer, scheduler, scaler, epoch, best_val_loss)
                print(f"         [*] New best val loss: {best_val_loss:.6f}")
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"]    = epoch + 1
            else:
                epochs_no_improve += 1

        # Broadcast early-stopping counter to all ranks
        if world_size > 1:
            stop_flag = torch.tensor(
                [epochs_no_improve if main else 0], device=device, dtype=torch.int
            )
            dist.broadcast(stop_flag, src=0)
            epochs_no_improve = stop_flag.item()

        if epochs_no_improve >= patience:
            if main:
                print(f"\n[INFO] Early stopping after {patience} epochs without improvement.")
            break

    if main:
        print("=" * 90)
        print(f"[INFO] Training complete. Best val loss: {best_val_loss:.6f}")
        print(f"[INFO] Best checkpoint: {best_ckpt}")

        # ---- Final test evaluation on rank 0 --------------------------------
        print("\n[INFO] Evaluating best model on test set …")
        raw_model = model.module if hasattr(model, "module") else model
        ckpt = torch.load(best_ckpt, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])

        test_loss, test_preds, test_tgts = run_epoch(
            model       = model,
            loader      = test_loader,
            device      = device,
            optimizer   = None,
            amp_dtype   = amp_dtype,
            use_amp     = use_amp,
            world_size  = 1,   # evaluate only on rank 0 with its full test split
        )
        if test_preds.shape[0] > 0:
            test_mae = mae_per_component(test_preds, test_tgts)
            print(f"[INFO] Test KL loss: {test_loss:.6f}")
            for i, name in enumerate(EVENTS):
                print(f"[INFO]   {name:12s} MAE: {test_mae[i].item():.4f}")
            if use_wandb:
                wandb.run.summary.update({
                    "test/kl_loss":          test_loss,
                    "test/mae_speciation":   test_mae[0].item(),
                    "test/mae_hgt":          test_mae[1].item(),
                    "test/mae_loss":         test_mae[2].item(),
                    "test/mae_duplication":  test_mae[3].item(),
                })

        if use_wandb:
            wandb.finish()

    # ---- Clean up -----------------------------------------------------------
    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Co-Phyloformer (DDP + BF16, JeanZay H100 ready)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    p.add_argument("--data_dir",       type=str, required=True,
                   help="Path to Datasets/ folder.")
    p.add_argument("--manifest_file",  type=str, default=None,
                   help="Pre-built file list (one .tgl path per line) for 1M+ datasets.")
    p.add_argument("--max_seq_len",    type=int, default=512,
                   help="Truncate / pad sequences to this length.")
    p.add_argument("--train_frac",     type=float, default=0.7)
    p.add_argument("--val_frac",       type=float, default=0.15)
    p.add_argument("--num_workers",    type=int, default=4,
                   help="DataLoader workers per GPU. 0 = main process only.")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Batches to prefetch per DataLoader worker.")

    # --- Training ---
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="Peak learning rate.")
    p.add_argument("--min_lr",          type=float, default=1e-6,
                   help="Minimum LR at end of cosine schedule.")
    p.add_argument("--weight_decay",    type=float, default=1e-2,
                   help="AdamW weight decay.")
    p.add_argument("--accumulate_grad", type=int,   default=8,
                   help="Gradient accumulation steps. Effective batch = world_size × this.")
    p.add_argument("--patience",        type=int,   default=15,
                   help="Early-stopping patience (epochs).")

    # --- Model ---
    p.add_argument("--d_model",                type=int,   default=128)
    p.add_argument("--nhead",                  type=int,   default=8)
    p.add_argument("--num_layers",             type=int,   default=2)
    p.add_argument("--dropout",                type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade compute for memory via activation checkpointing.")
    p.add_argument("--compile",                action="store_true",
                   help="Apply torch.compile() for extra speed on H100.")

    # --- Misc ---
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume",         type=str, default=None,
                   help="Path to checkpoint to resume from.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
