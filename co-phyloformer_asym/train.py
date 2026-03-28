
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from model import Cophyloformer
from data import CophylogenyDataset
from torch.nn import functional as F
import time
import numpy as np
import os
from plot import plot_event_metric_over_epochs, plot_epoch_loss_curve, plot_labels_vs_predictions
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from itertools import islice
import wandb
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric.strategies import DDPStrategy
from validation import run_full_validation, compute_val_predictions
import glob
import math

# CONFIGURATION FOR 1M DATASET (generalization run)

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

seed_everything(42)
event_names = [
    "Speciation",
    "HGT",
    "Loss",
    "Duplication",
]

start_time = time.time()  # Record start time


use_opm         = os.environ.get("USE_OPM", "0").strip() == "1"
use_dist_matrix = os.environ.get("USE_DIST_MATRIX", "0").strip() == "1"

class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir, mask_prob=0.1, pt_files=None):
        self.mask_prob = float(mask_prob)
        if pt_files is None:
            all_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
            valid_files = []
            for pt_path in all_files:
                try:
                    sample = torch.load(pt_path, map_location="cpu", weights_only=False)
                    if len(sample.get("host_msas", {})) == 0:
                        continue
                    if len(sample.get("parasite_msas", {})) == 0:
                        continue
                    valid_files.append(pt_path)
                except Exception:
                    # Corrupt/unreadable file -> skip deterministically
                    continue
            self.pt_files = valid_files
        else:
            self.pt_files = list(pt_files)
        self.preencoded_dir = preencoded_dir

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)

        def mask_sequence(sequence, mask_prob=0.1, mask_token=23, pad_token=22):
            if mask_prob <= 0:
                return sequence
            # Never mask PAD tokens: preserve padding semantics for attention/pooling masks.
            random_mask = torch.rand_like(sequence, dtype=torch.float32) < mask_prob
            random_mask &= (sequence != pad_token)
            masked = sequence.clone()
            masked[random_mask] = mask_token
            return masked

        # Should never happen thanks to pre-filtering in __init__
        if len(sample.get("host_msas", {})) == 0 or len(sample.get("parasite_msas", {})) == 0:
            raise ValueError(f"Invalid sample with empty MSAs: {pt_path}")

        host_list = list(sample["host_msas"].keys())
        parasite_list = list(sample["parasite_msas"].keys())
        
        parasite_idx_map = {name: i for i, name in enumerate(parasite_list)}
        host_idx_map = {name: i for i, name in enumerate(host_list)}
        valid_mappings = [
            (host_idx_map[h], parasite_idx_map[p])
            for p, h in sample["mappings"]
            if p in parasite_idx_map and h in host_idx_map
        ]

        labels = torch.tensor(
            [sample["event_frequencies"].get(event, 0.0) for event in event_names],
            dtype=torch.float32,
        )

        
        sim_time = torch.tensor([sample["event_frequencies"].get("Sim_time", 1.0)], dtype=torch.float32)
        out = {
            "host_msa": torch.stack([
                mask_sequence(encode_sequence(seq), mask_prob=self.mask_prob)
                for seq in sample["host_msas"].values()
            ]),
            "parasite_msa": torch.stack([
                mask_sequence(encode_sequence(seq), mask_prob=self.mask_prob)
                for seq in sample["parasite_msas"].values()
            ]),
            "mappings": valid_mappings,
            "labels": labels,
            "sim_time": sim_time,
        }
        if "host_dist" in sample:
            out["host_dist"] = sample["host_dist"]
            out["para_dist"] = sample["para_dist"]
        return out
def save_checkpoint(fabric, model, optimizer, lr_scheduler, epoch, val_loss, checkpoint_dir, filename, batch_idx=None, wandb_run_id=None, history=None):
    """Save model, optimizer, scheduler, and optional metric history via Fabric (DDP-safe, rank-0 only)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if batch_idx is not None:
        state["batch_idx"] = batch_idx
    if wandb_run_id is not None:
        state["wandb_run_id"] = wandb_run_id
    if history is not None:
        state["history"] = history
    fabric.save(os.path.join(checkpoint_dir, filename), state)


def load_checkpoint(fabric, model, optimizer, lr_scheduler, checkpoint_path):
    """Load checkpoint, handling both legacy raw-state-dict and Fabric-native formats."""
    probe = torch.load(checkpoint_path, map_location=fabric.device, weights_only=False)

    if "model_state_dict" in probe:
        # Legacy format — strip optional DDP 'module.' prefix and load directly
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in probe["model_state_dict"].items()}
        getattr(model, "module", model).load_state_dict(sd)
        if optimizer is not None and "optimizer_state_dict" in probe:
            optimizer.load_state_dict(probe["optimizer_state_dict"])
        if lr_scheduler is not None and "lr_scheduler" in probe:
            lr_scheduler.load_state_dict(probe["lr_scheduler"])
        meta = probe
    else:
        # Fabric-native format — let Fabric handle DDP-aware model/optimizer loading
        state = {"model": model, "optimizer": optimizer}
        meta = fabric.load(checkpoint_path, state)
        if lr_scheduler is not None and "lr_scheduler" in meta:
            lr_scheduler.load_state_dict(meta["lr_scheduler"])

    return (
        meta.get("epoch", 0),
        meta.get("val_loss", float("inf")),
        meta.get("batch_idx", None),
        meta.get("wandb_run_id", None),
        meta.get("history", None),
    )



def jukes_cantor_dist(msa: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Jukes-Cantor distances from a padded MSA token tensor.

    msa: [N, S]  int token ids  (PAD = 22)
    Returns: [N, N] float distance matrix (symmetric, zero diagonal).

    Fully vectorised — no Python loop over pairs.
    """
    valid = (msa != 22)                                      # [N, S]
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(0)     # [N, N, S]
    n_v = valid_pair.sum(dim=2).clamp(min=1).float()         # [N, N]
    mismatch = (msa.unsqueeze(1) != msa.unsqueeze(0)) & valid_pair  # [N, N, S]
    p = mismatch.float().sum(dim=2) / n_v                    # [N, N]
    p = p.clamp(0.0, 0.74)
    dist = -0.75 * torch.log(1.0 - (4.0 / 3.0) * p)
    dist.fill_diagonal_(0.0)
    return dist


def collate_fn(batch):
    """Pads MSA sequences dynamically to match batch size."""
    # After dataset pre-filtering, no element should be None.
    host_msas = [sample["host_msa"] for sample in batch]
    parasite_msas = [sample["parasite_msa"] for sample in batch]
    labels = torch.stack([sample["labels"] for sample in batch])
    mappings = [sample["mappings"] for sample in batch]  
    
    sim_time = torch.stack([sample["sim_time"] for sample in batch])

    #  Fix: Ensure consistent padding for batch processing 
    max_host_len = max(m.shape[0] for m in host_msas)
    max_parasite_len = max(m.shape[0] for m in parasite_msas)

    host_msas = [F.pad(m, (0, 0, 0, max_host_len - m.shape[0]), value=22) for m in host_msas]
    parasite_msas = [F.pad(m, (0, 0, 0, max_parasite_len - m.shape[0]), value=22) for m in parasite_msas]

    host_msas = torch.stack(host_msas)
    parasite_msas = torch.stack(parasite_msas)
    
    out = {
        "host_msa":     host_msas,
        "parasite_msa": parasite_msas,
        "labels":       labels,
        "mappings":     mappings,
        "sim_time":     sim_time,
    }

    if use_dist_matrix:
        if "host_dist" in batch[0]:
            # Precomputed distances available — pad [N,N] → [max_N, max_N] and stack
            def pad_dist(d, target):
                n = d.shape[0]
                if n == target:
                    return d
                out = torch.zeros(target, target, dtype=d.dtype)
                out[:n, :n] = d
                return out
            max_h = host_msas.shape[1]
            max_p = parasite_msas.shape[1]
            out["host_dist"] = torch.stack([pad_dist(s["host_dist"], max_h) for s in batch])
            out["para_dist"] = torch.stack([pad_dist(s["para_dist"], max_p) for s in batch])
        else:
            out["host_dist"] = torch.stack([jukes_cantor_dist(m) for m in host_msas])
            out["para_dist"] = torch.stack([jukes_cantor_dist(m) for m in parasite_msas])

    return out


def encode_sequence(sequence, max_len=200):
    """ Convert an MSA sequence string into a numerical tensor (simple one-hot encoding). """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"  # 21 tokens: 20 AAs + gap
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    UNK_ID = 21
    PAD_ID = 22

    encoded = [aa_to_index.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)

def main(fabric: Fabric):
    # Load Data
    preencoded_dir = "/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/test/"
    #preencoded_dir = '/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/test/'
    # Build file list once, then create train/val datasets with different masking policies.
    dataset = LazyCophyloformerDataset(preencoded_dir, mask_prob=0.0)
    # Train/Validation Split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )
    train_dataset = LazyCophyloformerDataset(
        preencoded_dir,
        mask_prob=0.15,
        pt_files=dataset.pt_files
    )
    val_dataset = LazyCophyloformerDataset(
        preencoded_dir,
        mask_prob=0.0,
        pt_files=dataset.pt_files
    )
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset   = torch.utils.data.Subset(val_dataset, val_indices)
    device = fabric.device
    epochs = 500

    batch_size = 8
    overfit_mode = os.environ.get("OVERFIT_MODE", "0").strip() == "1"
    train_num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "0" if overfit_mode else "8"))
    val_num_workers = int(os.environ.get("VAL_NUM_WORKERS", str(train_num_workers)))
    mid_epoch_validations = max(0, int(os.environ.get("MID_EPOCH_VALS", "2")))
    disable_checkpoints = os.environ.get("DISABLE_CHECKPOINTS", "0").strip() == "1"
    save_mid_epoch_val_ckpts = (not disable_checkpoints) and os.environ.get("SAVE_MID_EPOCH_VAL_CKPTS", "0").strip() == "1"
    collect_train_prediction_data = os.environ.get("COLLECT_TRAIN_PREDICTIONS", "0").strip() == "1"
    enable_wandb_watch = os.environ.get("WANDB_WATCH", "0").strip() == "1"
    gradient_checkpointing = os.environ.get("GRADIENT_CHECKPOINTING", "0").strip() == "1"

    # -----------------------------
    # Gradient accumulation
    # -----------------------------
    grad_accum_steps = 8

    # -----------------------------
    # DDP-safe sampling — let Fabric own the DistributedSampler so that
    # set_epoch() always targets the live sampler (not a replaced copy).
    # -----------------------------
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,   # Fabric replaces this with DistributedSampler(shuffle=True)
        collate_fn=collate_fn,
        num_workers=train_num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=val_num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    # After setup, Fabric has injected a DistributedSampler — reference it directly
    train_sampler = train_loader.sampler

    lr = 1e-4
    wd = 0.01

    # Asymmetric Huber: underprediction (target > pred) is penalized under_penalty times more.
    huber_delta = 1.0  # covers the full [0,1] label range quadratically
    under_penalty = 2.5

    def asymmetric_huber(pred, target):
        err = target - pred  # positive = underpredicting, negative = overpredicting
        abs_err = err.abs()
        loss = torch.where(
            abs_err < huber_delta,
            0.5 * abs_err ** 2,
            huber_delta * (abs_err - 0.5 * huber_delta),
        )
        weight = torch.where(err > 0,
                             torch.full_like(err, under_penalty),
                             torch.ones_like(err))
        return loss * weight

    criterion = asymmetric_huber  # used by validation calls
    event_loss_weights = torch.tensor([1.0, 2.0, 1.0, 1.0], device=device)
    tail_weight_scale = 1.0



    model = Cophyloformer(
        gradient_checkpointing=gradient_checkpointing,
        use_opm=use_opm,
        use_dist_matrix=use_dist_matrix,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    # Scheduler must be created before checkpoint loading so its state can be restored
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint logic
    checkpoint_dir = "checkpoints"
    resume_ckpt_path = os.environ.get("RESUME_CKPT", None)
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    wandb_run_id = None

    # History lists — initialised empty; overwritten from checkpoint on resume so
    # end-of-training plots always cover the full run, not just the resumed portion.
    epoch_losses = []
    mae_history = []
    mse_history = []
    mre_history = []
    smape_history = []
    val_mae_history = []
    val_mse_history = []
    val_mre_history = []
    val_smape_history = []
    val_loss_history = []

    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        if fabric.is_global_zero:
            print(f"[Checkpoint] Loading checkpoint from {resume_ckpt_path}")
        loaded_epoch, loaded_val_loss, loaded_batch_idx, loaded_run_id, loaded_history = load_checkpoint(
            fabric, model, optimizer, lr_scheduler, resume_ckpt_path
        )
        start_epoch = loaded_epoch
        start_batch = (loaded_batch_idx + 1) if loaded_batch_idx is not None else 0
        best_val_loss = loaded_val_loss
        wandb_run_id = loaded_run_id
        if loaded_history is not None:
            epoch_losses      = loaded_history.get("epoch_losses",      [])
            mae_history       = loaded_history.get("mae_history",       [])
            mse_history       = loaded_history.get("mse_history",       [])
            mre_history       = loaded_history.get("mre_history",       [])
            smape_history     = loaded_history.get("smape_history",     [])
            val_mae_history   = loaded_history.get("val_mae_history",   [])
            val_mse_history   = loaded_history.get("val_mse_history",   [])
            val_mre_history   = loaded_history.get("val_mre_history",   [])
            val_smape_history = loaded_history.get("val_smape_history", [])
            val_loss_history  = loaded_history.get("val_loss_history",  [])
            if fabric.is_global_zero:
                print(f"[Checkpoint] Restored metric history ({len(epoch_losses)} epochs)")
        if fabric.is_global_zero:
            print(f"[Checkpoint] Resumed from epoch {loaded_epoch}, best_val_loss {loaded_val_loss}, batch_idx {loaded_batch_idx}")

    entity = os.environ.get("WANDB_ENTITY", "cophylo_team")
    project = os.environ.get("WANDB_PROJECT", "CoPhyloformer")
    name_experiment = os.environ.get("WANDB_NAME", "1e-3NoWeightDecay")

    run = None
    run_id = None
    if fabric.global_rank == 0:
        # Dynamically extract model hyperparameters
        model_config = {}
        for attr in ["num_layers", "hidden_dim", "dropout", "embedding_dim", "num_heads"]:
            if hasattr(model, attr):
                model_config[attr] = getattr(model, attr)

        wandb_init_kwargs = dict(
            entity=entity,
            project=project,
            name=name_experiment,
            job_type="training",
            mode="offline",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": wd,
                "dataset_dir": preencoded_dir,
                "scheduler": "linear_warmup",
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "huber_delta": huber_delta,
                "event_loss_weights": event_loss_weights.tolist(),
                "tail_weight_scale": tail_weight_scale,
                "mid_epoch_validations": mid_epoch_validations,
                "save_mid_epoch_val_ckpts": save_mid_epoch_val_ckpts,
                "collect_train_prediction_data": collect_train_prediction_data,
                "gradient_checkpointing": gradient_checkpointing,
                "use_opm": use_opm,
                "use_dist_matrix": use_dist_matrix,
                "model_name": model.__class__.__name__,
                "dataset_size": len(dataset),
                **model_config
            },
        )
        if wandb_run_id is not None:
            # Resume the existing offline run: wandb writes a new local offline dir
            # with the same ID; `wandb sync` merges them on the server afterwards.
            wandb_init_kwargs["resume"] = "allow"
            wandb_init_kwargs["id"] = wandb_run_id

        # Initialize Weights & Biases
        run = wandb.init(**wandb_init_kwargs)
        run_id = run.id

        # Log total learnable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run.config["num_parameters"] = num_params

        # Watch gradients and parameters
        if enable_wandb_watch:
            wandb.watch(getattr(model, "module", model), log="all", log_freq=100)


    # Training loop over all batches per epoch (no micro-epochs)
    for epoch in range(start_epoch, epochs):
        # Ensure each epoch uses a different (but synchronized) shuffle order across ranks
        train_sampler.set_epoch(epoch)
        effective_steps_per_epoch = len(train_loader)
        # Number of optimizer (weight-update) steps this epoch
        effective_opt_steps_per_epoch = math.ceil(effective_steps_per_epoch / grad_accum_steps)
        # When resuming mid-epoch, start counting from the steps already completed
        if epoch == start_epoch and start_batch > 0:
            optimizer_step_count = start_batch // grad_accum_steps
        else:
            optimizer_step_count = 0
        # Trigger optional mid-epoch validation at evenly spaced optimizer steps.
        val_checkpoint_opt_steps = {
            int(math.ceil(k * effective_opt_steps_per_epoch / (mid_epoch_validations + 1)))
            for k in range(1, mid_epoch_validations + 1)
        }
        # Drop any checkpoints already passed at the resume point
        val_checkpoint_opt_steps = {s for s in val_checkpoint_opt_steps if s > optimizer_step_count}
        num_events = len(event_names)
        sum_abs_err = torch.zeros(num_events, device=device)
        sum_sq_err  = torch.zeros(num_events, device=device)
        sum_rel_err = torch.zeros(num_events, device=device)
        sum_smape   = torch.zeros(num_events, device=device)
        sample_count = 0
        all_train_prediction_data = [] if collect_train_prediction_data else None
        if fabric.global_rank == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}", leave=False):
            # Resume logic: skip earlier batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            elif epoch == start_epoch and batch_idx == start_batch:
                print(f"[Resume] Continuing from epoch {start_epoch+1}, batch {start_batch+1}")
            if batch is None:
                raise RuntimeError("collate_fn returned None; this would desync DDP ranks")
            batch["host_msa"] = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["sim_time"] = batch["sim_time"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            if use_dist_matrix:
                batch["host_dist"] = batch["host_dist"].to(device, non_blocking=True)
                batch["para_dist"] = batch["para_dist"].to(device, non_blocking=True)
            # Zero gradients only at the start of an accumulation window
            if (batch_idx % grad_accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
                host_dist=batch.get("host_dist"),
                para_dist=batch.get("para_dist"),
            )

            if collect_train_prediction_data:
                for idx in range(outputs.shape[0]):
                    row = {"Sample_Index": batch_idx * outputs.shape[0] + idx}
                    for i, event in enumerate(event_names):
                        row[f"{event}_Pred"] = outputs[idx, i].item()
                        row[f"{event}_GT"] = batch["labels"][idx, i].item()
                    all_train_prediction_data.append(row)

            # Asymmetric Huber: penalises underprediction more than overprediction.
            # Tail-aware per-sample weighting + per-task reweighting.
            target_weights = 1.0 + tail_weight_scale * batch["labels"]
            total_loss_tensor = sum(
                event_loss_weights[i] * (asymmetric_huber(outputs[:, i], batch["labels"][:, i]) * target_weights[:, i]).mean()
                for i in range(len(event_names))
            )

            loss_to_backprop = total_loss_tensor / grad_accum_steps
            
            fabric.backward(loss_to_backprop)

            # Perform optimizer step only when we have accumulated enough micro-batches
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if is_accum_step or is_last_batch:
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=0.5, error_if_nonfinite=False)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer_step_count += 1
                else:
                    if fabric.is_global_zero:
                        print(f"[Warning] Non-finite grad norm ({grad_norm:.2e}) at batch {batch_idx+1} — skipping update")
                    optimizer.zero_grad(set_to_none=True)

            current_step = batch_idx + 1
            if optimizer_step_count in val_checkpoint_opt_steps:
                # IMPORTANT: run validation on ALL ranks so any all_reduce/barrier inside
                # run_full_validation does not hang. Only rank0 logs/saves.
                val_results = run_full_validation(
                    fabric,
                    model,
                    val_loader,
                    criterion,
                    event_names,
                    device,
                    event_loss_weights=event_loss_weights,
                    tail_weight_scale=tail_weight_scale,
                )
                # Consume this opt-step so it cannot fire again on the next batch
                val_checkpoint_opt_steps.discard(optimizer_step_count)
                # DDP-consistent train metrics: average running accumulators across all ranks
                train_loss_step = fabric.all_reduce(
                    torch.tensor(total_loss / max(1, num_batches), device=device),
                    reduce_op="mean",
                ).item()
                global_sample_count = fabric.all_reduce(
                    torch.tensor(sample_count, device=device), reduce_op="sum"
                ).item()
                denom = max(1, global_sample_count)
                train_mae_step   = (fabric.all_reduce(sum_abs_err.clone(), reduce_op="sum") / denom).cpu().tolist()
                train_mse_step   = (fabric.all_reduce(sum_sq_err.clone(),  reduce_op="sum") / denom).cpu().tolist()
                train_mre_step   = (fabric.all_reduce(sum_rel_err.clone(), reduce_op="sum") / denom).cpu().tolist()
                train_smape_step = (fabric.all_reduce(sum_smape.clone(),   reduce_op="sum") / denom).cpu().tolist()
                if fabric.is_global_zero:
                    global_opt_step = epoch * effective_opt_steps_per_epoch + optimizer_step_count
                    pct_done = int(round(optimizer_step_count / effective_opt_steps_per_epoch * 100))
                    mae_str = " | ".join(
                        f"{event_names[i]}: {val_results['val_mae'][i]:.4f}"
                        for i in range(len(event_names))
                    )
                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"\n[Val {pct_done:3d}%] epoch {epoch+1}  "
                        f"opt-step {optimizer_step_count}/{effective_opt_steps_per_epoch}  "
                        f"lr: {current_lr:.2e}  "
                        f"val_loss: {val_results['val_loss']:.6f}  MAE → {mae_str}"
                    )
                    wandb.log({
                        "train/loss_step": train_loss_step,
                        "lr": optimizer.param_groups[0]['lr'],
                        "step": global_opt_step,
                        **{f"train/MAE_step/{event_names[i]}": train_mae_step[i]   for i in range(len(event_names))},
                        **{f"train/MSE_step/{event_names[i]}": train_mse_step[i]   for i in range(len(event_names))},
                        **{f"train/MRE_step/{event_names[i]}": train_mre_step[i]   for i in range(len(event_names))},
                        **{f"train/sMAPE_step/{event_names[i]}": train_smape_step[i] for i in range(len(event_names))},
                        "val/loss_step": val_results["val_loss"],
                        **{f"val/MAE_step/{event_names[i]}": val_results["val_mae"][i] for i in range(len(event_names))},
                        **{f"val/MSE_step/{event_names[i]}": val_results["val_mse"][i] for i in range(len(event_names))},
                        **{f"val/MRE_step/{event_names[i]}": val_results["val_mre"][i] for i in range(len(event_names))},
                        **{f"val/sMAPE_step/{event_names[i]}": val_results["val_smape"][i] for i in range(len(event_names))}
                    })
                    # History at mid-epoch contains all fully completed epochs so far
                    _mid_history = {
                        "epoch_losses":      epoch_losses,
                        "mae_history":       mae_history,
                        "mse_history":       mse_history,
                        "mre_history":       mre_history,
                        "smape_history":     smape_history,
                        "val_mae_history":   val_mae_history,
                        "val_mse_history":   val_mse_history,
                        "val_mre_history":   val_mre_history,
                        "val_smape_history": val_smape_history,
                        "val_loss_history":  val_loss_history,
                    }
                    ckpt_name = f"val_checkpoint_epoch{epoch+1}_step{current_step}.pth"
                    if save_mid_epoch_val_ckpts:
                        save_checkpoint(
                            fabric,
                            model,
                            optimizer,
                            lr_scheduler,
                            epoch,
                            val_results["val_loss"],
                            checkpoint_dir,
                            ckpt_name,
                            batch_idx=batch_idx,
                            wandb_run_id=run_id,
                            history=_mid_history,
                        )
                        print(f"[Checkpoint] Saved validation checkpoint: {ckpt_name}")
                    # --- Save BEST-OVERALL validation checkpoint (mid-epoch) ---
                    if val_results["val_loss"] < best_val_loss:
                        best_val_loss = val_results["val_loss"]
                        if save_mid_epoch_val_ckpts:
                            ckpt_name = f"best_overall_val_epoch{epoch+1}_step{current_step}.pth"
                            save_checkpoint(
                                fabric,
                                model,
                                optimizer,
                                lr_scheduler,
                                epoch,
                                best_val_loss,
                                checkpoint_dir,
                                ckpt_name,
                                batch_idx=batch_idx,
                                wandb_run_id=run_id,
                                history=_mid_history,
                            )
                        print(f"[Checkpoint] New BEST validation loss at step {current_step}: {best_val_loss:.6f}")

            total_loss += total_loss_tensor.item()
            num_batches += 1

            with torch.no_grad():
                preds = outputs.detach()
                labels = batch["labels"].detach()

                abs_err = (preds - labels).abs()
                sq_err  = (preds - labels).pow(2)
                labels_abs = labels.abs()

                rel_err = abs_err / (labels_abs + 1e-8)
                smape   = 2 * abs_err / (preds.abs() + labels_abs + 1e-8)


                sum_abs_err += abs_err.sum(dim=0)
                sum_sq_err  += sq_err.sum(dim=0)
                sum_rel_err += rel_err.sum(dim=0)
                sum_smape   += smape.sum(dim=0)
                sample_count += preds.shape[0]

        # Compute epoch loss on this rank
        epoch_loss = total_loss / max(1, num_batches)

        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        epoch_loss = fabric.all_reduce(epoch_loss_tensor, reduce_op="mean").item()

        epoch_losses.append(epoch_loss)

        if fabric.is_global_zero:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            print(f"Effective batch size: {batch_size * grad_accum_steps * fabric.world_size}")

        if fabric.is_global_zero:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }, step=(epoch + 1) * effective_opt_steps_per_epoch)

        #Synchronize metric accumulators across all ranks
        sum_abs_err = fabric.all_reduce(sum_abs_err, reduce_op="sum")
        sum_sq_err  = fabric.all_reduce(sum_sq_err,  reduce_op="sum")
        sum_rel_err = fabric.all_reduce(sum_rel_err, reduce_op="sum")
        sum_smape   = fabric.all_reduce(sum_smape,   reduce_op="sum")

        sample_count_tensor = torch.tensor(sample_count, device=device, dtype=torch.float32)
        sample_count = int(fabric.all_reduce(sample_count_tensor, reduce_op="sum").item())

        denom = max(sample_count, 1)
        mae = (sum_abs_err / denom).detach().cpu().tolist()
        mse = (sum_sq_err  / denom).detach().cpu().tolist()
        mre = (sum_rel_err / denom).detach().cpu().tolist()
        smape = (sum_smape / denom).detach().cpu().tolist()

        mae_history.append(mae)
        mse_history.append(mse)
        mre_history.append(mre)
        smape_history.append(smape)
    
        if fabric.is_global_zero:
            for i, event in enumerate(event_names):
                print(f"  {event}: MAE {mae[i]:.6f}, MSE {mse[i]:.6f}, MRE {mre[i]:.6f}, sMAPE {smape[i]:.6f}")
            # Print predictions only if at least one batch ran
            if 'outputs' in locals():
                for sample_idx in range(min(3, outputs.shape[0])):
                    print(f"\nEpoch {epoch+1}, Sample {sample_idx} - Predictions vs Ground Truth:")
                    for i, event in enumerate(event_names):
                        pred_val = preds[sample_idx, i].item()
                        gt_val = batch["labels"][sample_idx, i].item()
                        print(f"  {event}: Pred {pred_val:.4f}, GT {gt_val:.4f}")
            else:
                print(f"[Warning] No training batches completed in epoch {epoch+1}. Skipping sample preview.")
        # Log event-wise metrics to W&B
        if fabric.is_global_zero:
            metrics = {f"train/MAE/{event_names[i]}": mae[i] for i in range(len(event_names))}
            metrics.update({f"train/MSE/{event_names[i]}": mse[i] for i in range(len(event_names))})
            metrics.update({f"train/MRE/{event_names[i]}": mre[i] for i in range(len(event_names))})
            metrics.update({f"train/sMAPE/{event_names[i]}": smape[i] for i in range(len(event_names))})
            metrics["epoch"] = epoch + 1
            wandb.log(metrics, step=(epoch + 1) * effective_opt_steps_per_epoch)

        # VALIDATION PHASE replaced by function
        val_results = run_full_validation(
            fabric,
            model,
            val_loader,
            criterion,
            event_names,
            device,
            event_loss_weights=event_loss_weights,
            tail_weight_scale=tail_weight_scale,
        )
        val_loss = val_results["val_loss"]
        val_mae = val_results["val_mae"]
        val_mse = val_results["val_mse"]
        val_mre = val_results["val_mre"]
        val_smape = val_results["val_smape"]

        val_loss_history.append(val_loss)
        val_mae_history.append(val_mae)
        val_mse_history.append(val_mse)
        val_mre_history.append(val_mre)
        val_smape_history.append(val_smape)

        if fabric.is_global_zero:
            print(f"Validation Loss: {val_loss:.6f}")
            print("---- Validation Summary ----")
            print(f"Best Validation Loss (so far): {best_val_loss:.6f}")
            for i, event in enumerate(event_names):
                print(f"  {event}: val_MAE {val_mae[i]:.4f}, val_MSE {val_mse[i]:.4f}, val_MRE {val_mre[i]:.4f}, val_sMAPE {val_smape[i]:.4f}")
            print("----------------------------")
            wandb.log({
                "val/loss": val_loss,
                **{f"val/MAE/{event_names[i]}": val_mae[i] for i in range(len(event_names))},
                **{f"val/MSE/{event_names[i]}": val_mse[i] for i in range(len(event_names))},
                **{f"val/MRE/{event_names[i]}": val_mre[i] for i in range(len(event_names))},
                **{f"val/sMAPE/{event_names[i]}": val_smape[i] for i in range(len(event_names))},
                "epoch": epoch + 1,
            }, step=(epoch + 1) * effective_opt_steps_per_epoch)

        # Save end-of-epoch checkpoint (allows clean resume from the start of the next epoch)
        if fabric.is_global_zero and not disable_checkpoints:
            _history = {
                "epoch_losses":      epoch_losses,
                "mae_history":       mae_history,
                "mse_history":       mse_history,
                "mre_history":       mre_history,
                "smape_history":     smape_history,
                "val_mae_history":   val_mae_history,
                "val_mse_history":   val_mse_history,
                "val_mre_history":   val_mre_history,
                "val_smape_history": val_smape_history,
                "val_loss_history":  val_loss_history,
            }
            save_checkpoint(
                fabric,
                model,
                optimizer,
                lr_scheduler,
                epoch + 1,  # store next epoch so resume skips straight to it
                val_loss,
                checkpoint_dir,
                f"epoch_{epoch+1}_end.pth",
                batch_idx=None,
                wandb_run_id=run_id,
                history=_history,
            )
            print(f"[Checkpoint] Saved end-of-epoch checkpoint: epoch_{epoch+1}_end.pth")

            # Also update best checkpoint if end-of-epoch val is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    fabric,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch + 1,
                    best_val_loss,
                    checkpoint_dir,
                    f"best_overall_val_epoch{epoch+1}_end.pth",
                    batch_idx=None,
                    wandb_run_id=run_id,
                    history=_history,
                )
                print(f"[Checkpoint] New BEST validation loss at end of epoch {epoch+1}: {best_val_loss:.6f}")
        elif fabric.is_global_zero and val_loss < best_val_loss:
            best_val_loss = val_loss

        model.train()

    if fabric.is_global_zero:
        torch.save(model.state_dict(), "cophyloformer_custom_model.pth")

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Compute elapsed time

        print(f"Execution time: {elapsed_time:.4f} seconds")
        # --- Combined Train + Val LOSS curve ---
        plot_epoch_loss_curve(
            train_losses=epoch_losses,
            val_losses=val_loss_history,
            filename="combined_loss.png",
            use_log_scale=True
        )

        # --- Combined Train + Val METRIC curves (MAE, MSE, MRE, sMAPE) ---
        plot_event_metric_over_epochs(
            train_metrics=mae_history,
            val_metrics=val_mae_history,
            event_names=event_names,
            metric_name="MAE"
        )
        plot_event_metric_over_epochs(
            train_metrics=mse_history,
            val_metrics=val_mse_history,
            event_names=event_names,
            metric_name="MSE"
        )
        plot_event_metric_over_epochs(
            train_metrics=mre_history,
            val_metrics=val_mre_history,
            event_names=event_names,
            metric_name="MRE"
        )
        plot_event_metric_over_epochs(
            train_metrics=smape_history,
            val_metrics=val_smape_history,
            event_names=event_names,
            metric_name="sMAPE"
        )

        # Save final epoch predictions to a CSV file
        import csv

        output_path = "final_predictions.csv"

        # Compute full validation predictions properly
        val_preds_tensor, val_labels_tensor = compute_val_predictions(model, val_loader, device)

        fieldnames = ["Sample_Index"] + [f"{e}_Pred" for e in event_names] + [f"{e}_GT" for e in event_names]
        rows = []
        for i in range(len(val_preds_tensor)):
            row = {"Sample_Index": i}
            for j, event in enumerate(event_names):
                row[f"{event}_Pred"] = float(val_preds_tensor[i, j])
                row[f"{event}_GT"] = float(val_labels_tensor[i, j])
            rows.append(row)

        with open(output_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Final predictions saved to {output_path}")

        # --- Combined Train + Val Scatter Plots (Pred vs GT) ---
        # val_preds_tensor / val_labels_tensor already computed above — reuse them
        plot_images = {"plots/loss_curve": wandb.Image("combined_loss.png")}
        if collect_train_prediction_data and all_train_prediction_data:
            for j, event in enumerate(event_names):
                fname = f"combined_label_vs_pred_{event.lower()}.png"
                plot_labels_vs_predictions(
                    train_labels=[row[f"{event}_GT"] for row in all_train_prediction_data],
                    train_preds=[row[f"{event}_Pred"] for row in all_train_prediction_data],
                    val_labels=val_labels_tensor[:, j].numpy(),
                    val_preds=val_preds_tensor[:, j].numpy(),
                    event_name=event,
                    filename=fname,
                )
                plot_images[f"plots/labels_vs_preds_{event.lower()}"] = wandb.Image(fname)
        else:
            print("[Info] Skipping train-vs-val scatter plots (COLLECT_TRAIN_PREDICTIONS=0).")

        wandb.log(plot_images)

        # Save and log model as a W&B model artifact
        model_path = "cophyloformer_custom_model.pth"
        if os.path.exists(model_path):
            artifact = wandb.Artifact("cophyloformer", type="model")
            artifact.add_file(model_path)
            # Also include final predictions CSV as an associated file
            if os.path.exists("final_predictions.csv"):
                artifact.add_file("final_predictions.csv")
            run.log_artifact(artifact, aliases=["latest", f"epoch-{epochs}"])

        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=DDPStrategy(
            find_unused_parameters=True,
        )
    )
    fabric.launch(main)
