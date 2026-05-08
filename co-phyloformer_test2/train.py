
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
from plot import plot_event_metric_over_epochs, plot_epoch_loss_curve
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

# BEST CONFIGURATION SO FAR FOR SMALL DATASETS
# Log host switch 
#try new overfitting example again 

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')

seed_everything(42)
event_names = [
    "Speciation",
    "HGT",
    "Loss",
    "Duplication",
]

start_time = time.time()  # Record start time

class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir, mask_prob=0.1, pt_files=None):
        self.mask_prob = float(mask_prob)
        self.preencoded_dir = preencoded_dir
        if pt_files is None:
            self.pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
        else:
            self.pt_files = list(pt_files)

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

        if "host_msa" in sample and "para_msa" in sample:
            host_msa = sample["host_msa"].long()
            parasite_msa = sample["para_msa"].long()
            labels_dict = sample.get("labels", {})
            labels = torch.tensor(
                [labels_dict.get(event, 0.0) for event in event_names],
                dtype=torch.float32,
            )
            sim_time = torch.tensor([labels_dict.get("Sim_time", 1.0)], dtype=torch.float32)
            return {
                "host_msa": mask_sequence(host_msa, mask_prob=self.mask_prob),
                "parasite_msa": mask_sequence(parasite_msa, mask_prob=self.mask_prob),
                "mappings": sample.get("mappings", []),
                "labels": labels,
                "sim_time": sim_time,
            }

        if "host_msas" in sample and "parasite_msas" in sample:
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
            return {
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

        raise KeyError(
            f"Unsupported preencoded sample schema in {pt_path}. "
            f"Expected keys host_msa/para_msa/labels, got {sorted(sample.keys())}."
        )
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_dir, filename,
                    batch_idx=None, end_of_epoch=False, best_val_loss=None,
                    epoch_losses=None, val_loss_history=None):
    """Save model and optimizer state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss if best_val_loss is not None else val_loss,
        'end_of_epoch': end_of_epoch,
        'epoch_losses': epoch_losses if epoch_losses is not None else [],
        'val_loss_history': val_loss_history if val_loss_history is not None else [],
    }
    if batch_idx is not None:
        checkpoint['batch_idx'] = batch_idx
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, map_location=None):
    """Load model and optimizer state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    batch_idx = checkpoint.get('batch_idx', None)
    end_of_epoch = checkpoint.get('end_of_epoch', batch_idx is None)
    epoch_losses = checkpoint.get('epoch_losses', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    return epoch, val_loss, batch_idx, end_of_epoch, epoch_losses, val_loss_history

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
    
    return {
        "host_msa": host_msas,
        "parasite_msa": parasite_msas,
        "labels": labels,
        "mappings": mappings,  
        "sim_time": sim_time,  
    }


def encode_sequence(sequence, max_len=128):
    """ Convert an MSA sequence string into a numerical tensor (simple one-hot encoding). """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"  # 21 tokens: 20 AAs + gap
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    UNK_ID = 21
    PAD_ID = 22

    encoded = [aa_to_index.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)

def write_prediction_csv(preds, labels, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"true_{n}" for n in event_names] + [f"pred_{n}" for n in event_names])
        for true_row, pred_row in zip(labels.tolist(), preds.tolist()):
            writer.writerow(true_row + pred_row)

def main(fabric: Fabric, resume_ckpt_path=None):
    # Load Data
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_preencoded_dir = os.environ.get(
        "ASYMMETREE_TRAIN_PREENCODED_DIR",
        "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_train",
    )
    val_preencoded_dir = os.environ.get(
        "ASYMMETREE_VAL_PREENCODED_DIR",
        "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_val",
    )
    train_dataset = LazyCophyloformerDataset(train_preencoded_dir, mask_prob=0.1)
    val_dataset = LazyCophyloformerDataset(val_preencoded_dir, mask_prob=0.0)
    if fabric.global_rank == 0:
        print(
            f"Train preencoded dataset: dir={train_preencoded_dir!r}, "
            f"exists={os.path.isdir(train_preencoded_dir)}, pt_files={len(train_dataset)}",
            flush=True,
        )
        print(
            f"Validation preencoded dataset: dir={val_preencoded_dir!r}, "
            f"exists={os.path.isdir(val_preencoded_dir)}, pt_files={len(val_dataset)}",
            flush=True,
        )
    if len(train_dataset) < 1:
        raise ValueError(
            "Need at least 1 preencoded .pt sample for training. "
            f"Found {len(train_dataset)} in {train_preencoded_dir!r}. "
            f"Directory exists: {os.path.isdir(train_preencoded_dir)}. "
            "Check ASYMMETREE_TRAIN_PREENCODED_DIR and make sure it points to new_train."
        )
    if len(val_dataset) < 1:
        raise ValueError(
            "Need at least 1 preencoded .pt sample for validation. "
            f"Found {len(val_dataset)} in {val_preencoded_dir!r}. "
            f"Directory exists: {os.path.isdir(val_preencoded_dir)}. "
            "Check ASYMMETREE_VAL_PREENCODED_DIR and make sure it points to new_val."
        )
    device = fabric.device
    epochs = 20

    batch_size = 32

    # -----------------------------
    # Gradient accumulation
    # -----------------------------
    grad_accum_steps = 4

    # -----------------------------
    # DDP-safe sampling
    # -----------------------------
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=True,
        seed=42,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,  # IMPORTANT: do not use shuffle with a sampler
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=False,
    )

    # Create validation loader (also sharded for balanced work across ranks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    lr = 1e-4 # lower learning rate (5e-5, or 1e-5).
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
    # Keep the old MSA-transformer architecture, but train on AsymmeTree's four event frequencies.
    event_loss_weights = torch.ones(len(event_names), device=device)
    tail_weight_scale = 4.0

    model = Cophyloformer()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    # Scheduler should count *optimizer steps* (not micro-batches)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint logic
    checkpoint_dir = os.environ.get("CKPT_DIR", "checkpoints")
    resume_ckpt_path = resume_ckpt_path or os.environ.get("RESUME_CKPT", None)
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    epoch_losses = []
    val_loss_history = []
    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        if fabric.is_global_zero:
            print(f"[Checkpoint] Loading checkpoint from {resume_ckpt_path}")
        loaded_epoch, loaded_val_loss, loaded_batch_idx, loaded_end_of_epoch, loaded_epoch_losses, loaded_val_loss_history = load_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            resume_ckpt_path,
            map_location=fabric.device,
        )
        if loaded_end_of_epoch:
            start_epoch = loaded_epoch + 1
            start_batch = 0
        else:
            start_epoch = loaded_epoch
            start_batch = (loaded_batch_idx + 1) if loaded_batch_idx is not None else 0
        best_val_loss = loaded_val_loss
        epoch_losses = loaded_epoch_losses
        val_loss_history = loaded_val_loss_history
        if fabric.is_global_zero:
            print(
                f"[Checkpoint] Resumed from epoch {loaded_epoch}, best_val_loss {loaded_val_loss}, "
                f"batch_idx {loaded_batch_idx}, end_of_epoch {loaded_end_of_epoch}, "
                f"loss history: {len(epoch_losses)} epochs"
            )

    entity = os.environ.get("WANDB_ENTITY", "cophylo_team")
    project = os.environ.get("WANDB_PROJECT", "CoPhyloformer")
    name_experiment = os.environ.get("WANDB_NAME", "1e-3NoWeightDecay")
    mode = os.environ.get("WANDB_MODE", "online")  # "online", "offline", or "disabled"

    run = None
    if fabric.global_rank == 0:
        # Dynamically extract model hyperparameters
        model_config = {}
        for attr in ["num_layers", "hidden_dim", "dropout", "embedding_dim", "num_heads"]:
            if hasattr(model, attr):
                model_config[attr] = getattr(model, attr)

        # Initialize Weights & Biases
        run = wandb.init(
            entity=entity,
            project=project,
            name=name_experiment,
            job_type="training",
            config={
                "mode": mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": wd,
                "train_dataset_dir": train_preencoded_dir,
                "val_dataset_dir": val_preencoded_dir,
                "scheduler": "linear_warmup",
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "huber_delta": huber_delta,
                "event_loss_weights": event_loss_weights.tolist(),
                "tail_weight_scale": tail_weight_scale,
                "model_name": model.__class__.__name__,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                **model_config                 
            },
        )

        # Log total learnable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run.config["num_parameters"] = num_params

        # Watch gradients and parameters
        wandb.watch(getattr(model, "module", model), log="all", log_freq=100)


    mae_history = []
    mse_history = []
    mre_history = []
    smape_history = []
    val_mae_history = []
    val_mse_history = []
    val_mre_history = []
    val_smape_history = []
    # epoch_losses and val_loss_history initialized above (empty or restored from checkpoint)

    val_predictions_data = []
    best_step_predictions = None

    # Training loop over all batches per epoch (no micro-epochs)
    for epoch in range(start_epoch, epochs):
        # Ensure each epoch uses a different (but synchronized) shuffle order across ranks
        train_sampler.set_epoch(epoch)
        # One mid-epoch validation plus the existing end-of-epoch validation below.
        val_checkpoints = {max(1, math.ceil(len(train_loader) / 2))}
        num_events = len(event_names)
        sum_abs_err = torch.zeros(num_events, device=device)
        sum_sq_err  = torch.zeros(num_events, device=device)
        sum_rel_err = torch.zeros(num_events, device=device)
        sum_smape   = torch.zeros(num_events, device=device)
        sample_count = 0
        running_min_nonzero = torch.full((num_events,), float('inf'), device=device)
        all_train_prediction_data = []
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
            batch["host_msa"] = batch["host_msa"].to(device)
            batch["parasite_msa"] = batch["parasite_msa"].to(device)
            batch["sim_time"] = batch["sim_time"].to(device)
            batch["labels"] = batch["labels"].to(device)
            # Zero gradients only at the start of an accumulation window
            if (batch_idx % grad_accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
            )

            for idx in range(outputs.shape[0]):
                row = {"Sample_Index": batch_idx * outputs.shape[0] + idx}
                for event_idx, event in enumerate(event_names):
                    row[f"{event}_Pred"] = outputs[idx, event_idx].item()
                    row[f"{event}_GT"] = batch["labels"][idx, event_idx].item()
                all_train_prediction_data.append(row)

            # Asymmetric Huber: penalises underprediction more than overprediction.
            per_event_loss = asymmetric_huber(outputs, batch["labels"])
            target_weights = 1.0 + tail_weight_scale * batch["labels"]
            weighted_event_losses = (per_event_loss * target_weights).mean(dim=0)
            total_loss_tensor = (event_loss_weights * weighted_event_losses).sum()

            loss_to_backprop = total_loss_tensor / grad_accum_steps
            
            fabric.backward(loss_to_backprop)

            # Perform optimizer step only when we have accumulated enough micro-batches
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if is_accum_step or is_last_batch:
                fabric.clip_gradients(model, optimizer, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

            current_step = batch_idx + 1
            if current_step in val_checkpoints:
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
                if fabric.is_global_zero:
                    wandb.log({
                        "train/loss_step": total_loss_tensor.item(),
                        "lr": optimizer.param_groups[0]['lr'],
                        "step": epoch * (len(train_loader) // grad_accum_steps + (1 if (len(train_loader) % grad_accum_steps) else 0)) + (batch_idx // grad_accum_steps),
                        "val/loss_step": val_results["val_loss"],
                        **{f"val/MAE_step/{event_names[i]}": val_results["val_mae"][i] for i in range(len(event_names))},
                        **{f"val/MSE_step/{event_names[i]}": val_results["val_mse"][i] for i in range(len(event_names))},
                        **{f"val/MRE_step/{event_names[i]}": val_results["val_mre"][i] for i in range(len(event_names))},
                        **{f"val/sMAPE_step/{event_names[i]}": val_results["val_smape"][i] for i in range(len(event_names))}
                    })
                    # --- Save BEST-OVERALL validation checkpoint (mid-epoch) ---
                    if val_results["val_loss"] < best_val_loss:
                        best_val_loss = val_results["val_loss"]
                        val_predictions_data = val_results
                        best_ckpt_name = f"best_overall_val_epoch{epoch+1}_step{current_step}.pth"
                        save_checkpoint(
                            model,
                            optimizer,
                            lr_scheduler,
                            epoch,
                            best_val_loss,
                            checkpoint_dir,
                            best_ckpt_name,
                            batch_idx=batch_idx,
                            best_val_loss=best_val_loss,
                            epoch_losses=epoch_losses,
                            val_loss_history=val_loss_history,
                        )
                        print(f"[Checkpoint] New BEST validation loss at step {current_step}: {best_val_loss:.6f}")
                    ckpt_name = f"val_checkpoint_epoch{epoch+1}_step{current_step}.pth"
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        val_results["val_loss"],
                        checkpoint_dir,
                        ckpt_name,
                        batch_idx=batch_idx,
                        best_val_loss=best_val_loss,
                        epoch_losses=epoch_losses,
                        val_loss_history=val_loss_history,
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        val_results["val_loss"],
                        checkpoint_dir,
                        "latest.ckpt",
                        batch_idx=batch_idx,
                        best_val_loss=best_val_loss,
                        epoch_losses=epoch_losses,
                        val_loss_history=val_loss_history,
                    )
                    print(f"[Checkpoint] Saved validation checkpoint: {ckpt_name}")

            total_loss += total_loss_tensor.item()
            num_batches += 1

            with torch.no_grad():
                preds = outputs.detach()
                labels = batch["labels"].detach()

                abs_err = (preds - labels).abs()
                sq_err  = (preds - labels).pow(2)
                labels_abs = labels.abs()

                safe_labels = torch.where(labels_abs > 0, labels_abs, torch.full_like(labels_abs, float('inf')))
                batch_min = torch.amin(safe_labels, dim=0)
                running_min_nonzero = torch.minimum(running_min_nonzero, batch_min)

                fallback_eps = torch.finfo(labels.dtype).eps
                eps_vec = torch.where(torch.isfinite(running_min_nonzero), running_min_nonzero * 1e-2, torch.full_like(running_min_nonzero, fallback_eps))

                rel_err = abs_err / (labels_abs + eps_vec)
                smape   = 2 * abs_err / (preds.abs() + labels_abs + eps_vec)

                current_step = batch_idx + 1
                if current_step in val_checkpoints:
                    batch_mae   = abs_err.mean(dim=0).detach().cpu().tolist()
                    batch_mse   = sq_err.mean(dim=0).detach().cpu().tolist()
                    batch_mre   = rel_err.mean(dim=0).detach().cpu().tolist()
                    batch_smape = smape.mean(dim=0).detach().cpu().tolist()

                    if fabric.is_global_zero:
                        log_dict = {
                            "step": epoch * (len(train_loader) // grad_accum_steps + (1 if (len(train_loader) % grad_accum_steps) else 0)) + (batch_idx // grad_accum_steps),
                            "lr": optimizer.param_groups[0]['lr']
                        }
                        for i, event in enumerate(event_names):
                            log_dict[f"MAE_step/{event}"] = batch_mae[i]
                            log_dict[f"MSE_step/{event}"] = batch_mse[i]
                            log_dict[f"MRE_step/{event}"] = batch_mre[i]
                            log_dict[f"sMAPE_step/{event}"] = batch_smape[i]

                        wandb.log(log_dict)

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
                
            })

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
            metrics = {f"MAE/{event_names[i]}": mae[i] for i in range(len(event_names))}
            metrics.update({f"MSE/{event_names[i]}": mse[i] for i in range(len(event_names))})
            metrics.update({f"MRE/{event_names[i]}": mre[i] for i in range(len(event_names))})
            metrics.update({f"sMAPE/{event_names[i]}": smape[i] for i in range(len(event_names))})
            metrics["epoch"] = epoch + 1
            wandb.log(metrics)

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    best_val_loss,
                    checkpoint_dir,
                    f"best_overall_val_epoch{epoch+1}_end.pth",
                    end_of_epoch=True,
                    best_val_loss=best_val_loss,
                    epoch_losses=epoch_losses,
                    val_loss_history=val_loss_history,
                )
                print(f"[Checkpoint] New BEST validation loss at epoch end: {best_val_loss:.6f}")
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                val_loss,
                checkpoint_dir,
                "latest.ckpt",
                end_of_epoch=True,
                best_val_loss=best_val_loss,
                epoch_losses=epoch_losses,
                val_loss_history=val_loss_history,
            )
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                val_loss,
                checkpoint_dir,
                "last_epoch.ckpt",
                end_of_epoch=True,
                best_val_loss=best_val_loss,
                epoch_losses=epoch_losses,
                val_loss_history=val_loss_history,
            )
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
            })

        model.train()

    if fabric.is_global_zero:
        torch.save(model.state_dict(), "cophyloformer_custom_model.pth")

        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")

        # --- Loss curve (train + val over all epochs, including resumed history) ---
        plot_epoch_loss_curve(
            train_losses=epoch_losses,
            val_losses=val_loss_history,
            filename="combined_loss.png",
            use_log_scale=True
        )
        wandb.log({"plots/loss_curve": wandb.Image("combined_loss.png")})

        # --- Save train + val prediction CSVs for post_process.py ---
        val_preds_tensor, val_labels_tensor = compute_val_predictions(model, val_loader, device)

        train_preds_tensor  = torch.tensor(
            [[row[f"{e}_Pred"] for e in event_names] for row in all_train_prediction_data]
        )
        train_labels_tensor = torch.tensor(
            [[row[f"{e}_GT"]   for e in event_names] for row in all_train_prediction_data]
        )
        train_csv = os.path.join(checkpoint_dir, "train_predictions.csv")
        val_csv   = os.path.join(checkpoint_dir, "val_predictions.csv")
        write_prediction_csv(train_preds_tensor, train_labels_tensor, train_csv)
        write_prediction_csv(val_preds_tensor,   val_labels_tensor,   val_csv)
        print(f"Predictions saved to {train_csv} / {val_csv}")
        print(f"Run post_process.py --output-dir . --train-csv {train_csv} --val-csv {val_csv}")

        # Save and log model artifact
        model_path = "cophyloformer_custom_model.pth"
        if os.path.exists(model_path):
            artifact = wandb.Artifact("cophyloformer", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact, aliases=["latest", f"epoch-{epochs}"])

        wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train Co-Phyloformer")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", description="Train from scratch")
    resumer = subparsers.add_parser("resume", description="Resume from a checkpoint")
    resumer.add_argument("checkpoint", help="Path to .ckpt or .pth checkpoint")
    args = parser.parse_args()

    resume_ckpt_path = args.checkpoint if args.command == "resume" else None

    def _main(fabric: Fabric):
        main(fabric, resume_ckpt_path=resume_ckpt_path)

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=DDPStrategy(
            find_unused_parameters=True,
        )
    )
    fabric.launch(_main)
