import csv
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
from model import Cophyloformer
from validation import run_full_validation
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric.strategies import DDPStrategy

# train plot on top of validation or use density plot to see better the results 

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch._dynamo.config.optimize_ddp = False  # flex_attention uses higher-order ops incompatible with DDP optimizer

seed_everything(42)

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]
TRAIN_PRED_MAX = 2000  # samples collected per epoch for CSV/plots


def write_prediction_csv(preds, labels, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"true_{n}" for n in EVENT_NAMES] + [f"pred_{n}" for n in EVENT_NAMES])
        for true_row, pred_row in zip(labels.tolist(), preds.tolist()):
            writer.writerow(true_row + pred_row)

# --- env flags ---
use_opm         = os.environ.get("USE_OPM", "0").strip() == "1"
axial_layers     = int(os.environ.get("AXIAL_LAYERS", "2"))
cross_layers     = int(os.environ.get("CROSS_LAYERS", "1"))
hidden_dim       = int(os.environ.get("HIDDEN_DIM", "256"))
grad_ckpt        = os.environ.get("GRADIENT_CHECKPOINTING", "0").strip() == "1"
use_flex         = os.environ.get("USE_FLEX_ATTENTION", "0").strip() == "1"  # requires PyTorch >= 2.5
use_compile      = os.environ.get("USE_COMPILE", "0").strip() == "1"
host_max_leaves  = int(os.environ.get("HOST_MAX_LEAVES", "51"))  # host trees have max 50 leaves
para_max_leaves  = int(os.environ.get("PARA_MAX_LEAVES", "142"))  # parasite trees have max 128, avg 82; cap for memory
dropout          = float(os.environ.get("DROPOUT", "0.1"))
use_bucketed_batches = os.environ.get("USE_BUCKETED_BATCHES", "1").strip() == "1"
bucket_size      = int(os.environ.get("BUCKET_SIZE", "8"))

PAD_ID = 22
BUCKET_META_NAME = "bucket_meta.tsv"

def unwrap_model(model):
    """Safely unwrap Fabric and DDP wrappers to get the base PyTorch model."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m

class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir, pt_files=None):
        if pt_files is None:
            self.pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
        else:
            self.pt_files = list(pt_files)
        self.bucket_keys = None

    def __len__(self):
        return len(self.pt_files)

    def _bucket_meta_path(self):
        return os.path.join(os.path.dirname(self.pt_files[0]), BUCKET_META_NAME) if self.pt_files else None

    def _load_bucket_keys_from_metadata(self):
        meta_path = self._bucket_meta_path()
        if meta_path is None or not os.path.exists(meta_path):
            return False

        sizes = {}
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sizes[row["file"]] = (
                    int(row["host_leaves"]),
                    int(row["para_leaves"]),
                    int(row["mapping_count"]),
                )

        keys = []
        for pt_path in self.pt_files:
            meta = sizes.get(os.path.basename(pt_path))
            if meta is None:
                return False
            host_n, para_n, mapping_n = meta
            keys.append((
                min(para_n, para_max_leaves) // bucket_size,
                min(host_n, host_max_leaves) // bucket_size,
                mapping_n // bucket_size,
            ))
        self.bucket_keys = keys
        return True

    def build_bucket_keys(self):
        if self._load_bucket_keys_from_metadata():
            return

        keys = []
        for pt_path in tqdm(self.pt_files, desc=f"Bucket scan {os.path.basename(os.path.normpath(os.path.dirname(pt_path)))}", leave=False):
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
            host_n = min(int(sample["host_msa"].shape[0]), host_max_leaves)
            para_n = min(int(sample["para_msa"].shape[0]), para_max_leaves)
            mapping_n = int(len(sample["mappings"]))
            keys.append((para_n // bucket_size, host_n // bucket_size, mapping_n // bucket_size))
        self.bucket_keys = keys

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx], map_location="cpu", weights_only=False)
        labels_src = sample["labels"]
        return {
            "host_msa":     sample["host_msa"],
            "parasite_msa": sample["para_msa"],
            "mappings":     sample["mappings"],
            "labels": torch.tensor(
                [labels_src.get(e, 0.0) for e in EVENT_NAMES],
                dtype=torch.float32,
            ),
            "sim_time": torch.tensor(
                [labels_src.get("Sim_time", 1.0)],
                dtype=torch.float32,
            ),
            "host_dist": sample["host_dist"],
            "para_dist":  sample["para_dist"],
        }


class BucketedDistributedBatchSampler:
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, drop_last=False, seed=42):
        if dataset.bucket_keys is None:
            raise ValueError("BucketedDistributedBatchSampler requires dataset.bucket_keys")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        buckets = {}
        for idx, key in enumerate(self.dataset.bucket_keys):
            buckets.setdefault(key, []).append(idx)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_items = list(buckets.items())
        if self.shuffle and len(bucket_items) > 1:
            order = torch.randperm(len(bucket_items), generator=g).tolist()
            bucket_items = [bucket_items[i] for i in order]

        batches = []
        for _, indices in bucket_items:
            if self.shuffle and len(indices) > 1:
                order = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]

        if batches:
            remainder = len(batches) % self.num_replicas
            if remainder:
                batches.extend(batches[:self.num_replicas - remainder])
        return batches

    def __iter__(self):
        batches = self._global_batches()
        yield from batches[self.rank::self.num_replicas]

    def __len__(self):
        batches = self._global_batches()
        return len(batches[self.rank::self.num_replicas])


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        print("[Warning] all samples in batch were invalid, returning None")
        return None

    def pad(msas, cap, pad_val=PAD_ID):
        max_n = min(max(m.shape[0] for m in msas), cap)
        return torch.stack([F.pad(m[:max_n], (0, 0, 0, max(0, max_n - m.shape[0])), value=pad_val) for m in msas])

    host_msas = pad([s["host_msa"] for s in batch], host_max_leaves)
    para_msas = pad([s["parasite_msa"] for s in batch], para_max_leaves)
    def pad_dist(d, n):
        if d.shape[0] == n:
            return d
        out = torch.zeros(n, n, dtype=d.dtype)
        out[:d.shape[0], :d.shape[0]] = d
        return out
    return {
        "host_msa":     host_msas,
        "parasite_msa": para_msas,
        "labels":       torch.stack([s["labels"] for s in batch]),
        "mappings":     [s["mappings"] for s in batch],
        "sim_time":     torch.stack([s["sim_time"] for s in batch]),
        "host_dist":    torch.stack([pad_dist(s["host_dist"], host_msas.shape[1]) for s in batch]),
        "para_dist":    torch.stack([pad_dist(s["para_dist"], para_msas.shape[1]) for s in batch]),
    }


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
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
            new_best_val = min(best_val, val_loss)
            state = {
                "model":      unwrap_model(model).state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "step":       global_step,
                "epoch_step": 0 if end_of_epoch else epoch_step,
                "val_loss":   val_loss,
                "best_val":   new_best_val,
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
    The scheduler is fast-forwarded to the correct step rather than restoring
    its saved state, so it stays consistent with the freshly-computed schedule.
    """
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # Fast-forward the scheduler instead of restoring saved state.
    # Restoring state can misalign progress if total_steps changed between runs.
    global_step = ckpt["step"] + 1
    for _ in range(global_step):
        scheduler.step()
    start_epoch = ckpt["epoch"]
    epoch_step  = ckpt.get("epoch_step", 0)
    best_val    = ckpt.get("best_val", ckpt["val_loss"])
    if fabric.is_global_zero:
        print(f"[Resume] epoch={start_epoch}  step={global_step}  "
              f"lr={optimizer.param_groups[0]['lr']:.3e}  best_val={best_val:.6f}")
    return start_epoch, global_step, epoch_step, best_val


def main(fabric: Fabric, ckpt_to_load=None):
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    train_preencoded_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_train"
    val_preencoded_dir   = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_val"
    epochs         = int(os.environ.get("EPOCHS", "50"))
    batch_size     = int(os.environ.get("BATCH_SIZE", "64"))  # per GPU
    grad_accum     = int(os.environ.get("GRAD_ACCUM", "4"))
    lr             = float(os.environ.get("LR", "1e-4"))
    wd             = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    huber_delta    = float(os.environ.get("HUBER_DELTA", "1.0"))
    under_penalty  = float(os.environ.get("UNDER_PENALTY", "2.5"))
    tail_weight    = float(os.environ.get("TAIL_WEIGHT", "2.0"))
    hgt_loss_weight = float(os.environ.get("HGT_LOSS_WEIGHT", "2.0"))
    mid_epoch_vals = int(os.environ.get("MID_EPOCH_VALS", "1"))
    num_workers    = int(os.environ.get("NUM_WORKERS", "8"))
    ckpt_dir       = os.environ.get("CKPT_DIR", "checkpoints")
    device         = fabric.device

    hparams = {
        "train_preencoded_dir": train_preencoded_dir,
        "val_preencoded_dir":   val_preencoded_dir,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "grad_accum":      grad_accum,
        "lr":              lr,
        "wd":              wd,
        "huber_delta":     huber_delta,
        "under_penalty":      under_penalty,
        "tail_weight":        tail_weight,
        "hgt_loss_weight":    hgt_loss_weight,
        "dropout":            dropout,
        "axial_layers":       axial_layers,
        "use_opm":            use_opm,
        "use_flexattention":  use_flex,
        "use_compile":        use_compile,
        "use_bucketed_batches": use_bucketed_batches,
        "bucket_size":        bucket_size,
        "pair_dim":           64,
        "cls_dim":            512,
        "mid_epoch_vals":  mid_epoch_vals,
        "ckpt_dir":        ckpt_dir,
    }

    event_loss_weights = torch.tensor([1.0, hgt_loss_weight, 1.0, 1.0], device=device)

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
    train_dataset = LazyCophyloformerDataset(train_preencoded_dir)
    val_dataset   = LazyCophyloformerDataset(val_preencoded_dir)

    if use_bucketed_batches:
        if fabric.is_global_zero:
            train_dataset.build_bucket_keys()
            val_dataset.build_bucket_keys()
        fabric.barrier()
        if not fabric.is_global_zero:
            train_dataset.build_bucket_keys()
            val_dataset.build_bucket_keys()

        train_sampler = BucketedDistributedBatchSampler(
            train_dataset, batch_size=batch_size,
            num_replicas=fabric.world_size, rank=fabric.global_rank,
            shuffle=True, seed=42,
        )
        val_sampler = BucketedDistributedBatchSampler(
            val_dataset, batch_size=batch_size * 2,
            num_replicas=fabric.world_size, rank=fabric.global_rank,
            shuffle=False, seed=42,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  collate_fn=collate_fn, num_workers=num_workers,
                                  persistent_workers=num_workers > 0,
                                  prefetch_factor=4 if num_workers > 0 else None, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                collate_fn=collate_fn, num_workers=num_workers,
                                persistent_workers=num_workers > 0,
                                prefetch_factor=2 if num_workers > 0 else None, pin_memory=True)
    else:
        # batch_size is per-GPU — use manual DistributedSampler so Fabric doesn't
        # divide it again by num_processes.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=False,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, num_workers=num_workers,
                                  persistent_workers=num_workers > 0,
                                  prefetch_factor=4 if num_workers > 0 else None, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size * 2, sampler=val_sampler,
                                  collate_fn=collate_fn, num_workers=num_workers,
                                  persistent_workers=num_workers > 0,
                                  prefetch_factor=2 if num_workers > 0 else None, pin_memory=True)
    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader, use_distributed_sampler=False
    )

    # -------------------------------------------------------------------------
    # Model + optimizer + scheduler
    # -------------------------------------------------------------------------
    model = Cophyloformer(
        hidden_dim=hidden_dim, pair_dim=64, cls_dim=512, axial_layers=axial_layers,
        use_opm=use_opm, use_dist_matrix=True,
        gradient_checkpointing=grad_ckpt,
        num_cross_layers=cross_layers,
        use_flexattention=use_flex,
        dropout=dropout,
    )
    if use_compile:
        model = torch.compile(model, dynamic=True)
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
        run_name = os.environ.get("WANDB_NAME", "run")
        if start_epoch > 0:
            run_name = f"{run_name}_resume_ep{start_epoch}"
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "cophylo_team"),
            project=os.environ.get("WANDB_PROJECT", "CoPhyloformer"),
            name=run_name,
            group=os.environ.get("WANDB_NAME", "run"),
            id=wandb.util.generate_id(),
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={
                **hparams,
                "total_steps":  total_steps,
                "warmup_steps": warmup_steps,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
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
                       **{f"val/MAE_mid/{EVENT_NAMES[i]}": vr["val_mae"][i] for i in range(len(EVENT_NAMES))},
                       **{f"val/MRE_mid/{EVENT_NAMES[i]}": vr["val_mre"][i] for i in range(len(EVENT_NAMES))}},
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

        total_loss    = 0.0
        num_batches   = 0
        epoch_step    = 0
        skipped_steps = 0

        sum_abs   = torch.zeros(len(EVENT_NAMES), device=device)
        sum_rel   = torch.zeros(len(EVENT_NAMES), device=device)
        n_samples = 0

        train_pred_list   = []
        train_label_list  = []
        train_pred_count  = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                     desc=f"Epoch {epoch+1}", leave=False):
            if batch is None:
                continue
            batch["host_msa"]     = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["sim_time"]     = batch["sim_time"].to(device, non_blocking=True)
            batch["labels"]       = batch["labels"].to(device, non_blocking=True)
            batch["host_dist"]    = batch["host_dist"].to(device, non_blocking=True)
            batch["para_dist"]    = batch["para_dist"].to(device, non_blocking=True)

            if batch_idx % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            is_accum = ((batch_idx + 1) % grad_accum == 0)
            is_last  = ((batch_idx + 1) == len(train_loader))
            sync_gradients = is_accum or is_last

            with fabric.no_backward_sync(model, enabled=not sync_gradients):
                outputs = model(
                    batch["host_msa"], batch["parasite_msa"], batch["mappings"],
                    batch["sim_time"],
                    host_dist=batch["host_dist"], para_dist=batch["para_dist"],
                )

                tw = 1.0 + tail_weight * batch["labels"]
                loss = sum(
                    event_loss_weights[i] * (asymmetric_huber(outputs[:, i], batch["labels"][:, i]) * tw[:, i]).mean()
                    for i in range(len(EVENT_NAMES))
                )
                fabric.backward(loss / grad_accum)

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
                    skipped_steps += 1
                    global_step   += 1  # always advance so validation still triggers
                    if fabric.is_global_zero:
                        print(f"[Warning] non-finite grad norm {gnorm:.2e} at batch {batch_idx+1}, skipping")
                    optimizer.zero_grad(set_to_none=True)

            total_loss  += loss.item()
            num_batches += 1
            with torch.no_grad():
                diff       = (outputs.detach() - batch["labels"]).abs()
                sum_abs   += diff.sum(dim=0)
                sum_rel   += (diff / batch["labels"].clamp(min=0.01)).sum(dim=0)
                n_samples += outputs.shape[0]
                if fabric.is_global_zero and train_pred_count < TRAIN_PRED_MAX:
                    take = min(outputs.shape[0], TRAIN_PRED_MAX - train_pred_count)
                    train_pred_list.append(outputs.detach()[:take].cpu())
                    train_label_list.append(batch["labels"][:take].cpu())
                    train_pred_count += take

        # --- End of epoch ---
        epoch_loss = fabric.all_reduce(torch.tensor(total_loss / max(1, num_batches), device=device),
                                       reduce_op="mean").item()
        global_abs = fabric.all_reduce(sum_abs, reduce_op="sum")
        global_rel = fabric.all_reduce(sum_rel, reduce_op="sum")
        global_n   = int(fabric.all_reduce(torch.tensor(n_samples, device=device), reduce_op="sum").item())
        mae = (global_abs / max(1, global_n)).cpu().tolist()
        mre = (global_rel / max(1, global_n)).cpu().tolist()

        if fabric.is_global_zero:
            print(f"  train loss={epoch_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {mae[i]:.4f}" for i in range(len(EVENT_NAMES))))
            if skipped_steps > 0:
                print(f"  [Warning] {skipped_steps}/{epoch_step + skipped_steps} optimizer steps skipped (non-finite gradients)")

        # end-of-epoch validation + checkpoint
        fabric.barrier()
        vr = run_full_validation(fabric, model, val_loader, asymmetric_huber,
                                  EVENT_NAMES, device,
                                  event_loss_weights=event_loss_weights,
                                  tail_weight_scale=tail_weight,
                                  collect_preds=True)
        fabric.barrier()
        val_loss = vr["val_loss"]
        val_mae  = vr["val_mae"]
        val_mre  = vr["val_mre"]

        if fabric.is_global_zero and train_pred_list and "val_preds" in vr:
            train_preds  = torch.cat(train_pred_list)
            train_labels = torch.cat(train_label_list)
            write_prediction_csv(train_preds, train_labels,
                                 os.path.join(ckpt_dir, "train_predictions.csv"))
            write_prediction_csv(vr["val_preds"], vr["val_labels"],
                                 os.path.join(ckpt_dir, "val_predictions.csv"))
            print(f"  [CSV] predictions written to {ckpt_dir}/")

        if fabric.is_global_zero:
            print(f"  val  loss={val_loss:.6f}")
            print("  MAE  " + "  ".join(f"{EVENT_NAMES[i]}: {val_mae[i]:.4f}" for i in range(len(EVENT_NAMES))))
            print("  MRE  " + "  ".join(f"{EVENT_NAMES[i]}: {val_mre[i]:.4f}" for i in range(len(EVENT_NAMES))))
            ep_step = (epoch + 1) * opt_steps_per_epoch
            wandb.log({
                "epoch": epoch + 1, "train/loss": epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/MAE/{EVENT_NAMES[i]}": mae[i] for i in range(len(EVENT_NAMES))},
                **{f"train/MRE/{EVENT_NAMES[i]}": mre[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step)
            wandb.log({
                "epoch": epoch + 1, "val/loss": val_loss,
                **{f"val/MAE/{EVENT_NAMES[i]}": val_mae[i] for i in range(len(EVENT_NAMES))},
                **{f"val/MRE/{EVENT_NAMES[i]}": val_mre[i] for i in range(len(EVENT_NAMES))},
            }, step=ep_step + 1)

        best_val = save_checkpoint(
            fabric, model, optimizer, scheduler,
            epoch, global_step, epoch_step, val_loss,
            hparams, ckpt_dir, best_val, end_of_epoch=True,
        )

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------
    if fabric.is_global_zero:
        torch.save(unwrap_model(model).state_dict(), "cophyloformer_final.pth")
        wandb.finish()
        print("Training complete.")
        print("Run post_process.py on CPU to generate scatter plots and prediction CSV files.")


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
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    fabric.launch(_main)
