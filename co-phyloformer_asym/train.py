
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
            "sim_time": sim_time  
        }
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


def encode_sequence(sequence, max_len=500):
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
    preencoded_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/asym_preencoded/"
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
    epochs = 20

    batch_size = 8

    # -----------------------------
    # Gradient accumulation
    # -----------------------------
    grad_accum_steps = 8

    # -----------------------------
    # DDP-safe sampling
    # -----------------------------
    train_sampler = DistributedSampler(
        train_subset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=True,
        seed=42,
    )

    val_sampler = DistributedSampler(
        val_subset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_subset,
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
        val_subset,
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

    lr = 3e-4
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

    model = Cophyloformer(gradient_checkpointing=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    # Scheduler must be created before checkpoint loading so its state can be restored
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 2000

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
        wandb.watch(getattr(model, "module", model), log="all", log_freq=100)


    # Training loop over all batches per epoch (no micro-epochs)
    for epoch in range(start_epoch, epochs):
        # Ensure each epoch uses a different (but synchronized) shuffle order across ranks
        train_sampler.set_epoch(epoch)
        effective_steps_per_epoch = len(train_loader)
        # Number of optimizer (weight-update) steps this epoch
        effective_opt_steps_per_epoch = math.ceil(effective_steps_per_epoch / grad_accum_steps)
        # Trigger validation at every 10 % of *optimizer* steps (10 %, 20 %, … 100 %)
        # Stored as the batch_idx+1 value at which the corresponding opt-step fires.
        val_checkpoint_opt_steps = {
            int(math.ceil(pct / 10 * effective_opt_steps_per_epoch))
            for pct in range(1, 11)          # 1..10 → 10 % … 100 %
        }
        optimizer_step_count = 0
        num_events = len(event_names)
        sum_abs_err = torch.zeros(num_events, device=device)
        sum_sq_err  = torch.zeros(num_events, device=device)
        sum_rel_err = torch.zeros(num_events, device=device)
        sum_smape   = torch.zeros(num_events, device=device)
        sample_count = 0
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
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=1.0, error_if_nonfinite=False)
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
                if fabric.is_global_zero:
                    global_opt_step = epoch * effective_opt_steps_per_epoch + optimizer_step_count
                    pct_done = int(round(optimizer_step_count / effective_opt_steps_per_epoch * 100))
                    mae_str = " | ".join(
                        f"{event_names[i]}: {val_results['val_mae'][i]:.4f}"
                        for i in range(len(event_names))
                    )
                    print(
                        f"\n[Val {pct_done:3d}%] epoch {epoch+1}  "
                        f"opt-step {optimizer_step_count}/{effective_opt_steps_per_epoch}  "
                        f"val_loss: {val_results['val_loss']:.6f}  MAE → {mae_str}"
                    )
                    wandb.log({
                        "train/loss_step": total_loss_tensor.item(),
                        "lr": optimizer.param_groups[0]['lr'],
                        "step": global_opt_step,
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

                if optimizer_step_count in val_checkpoint_opt_steps:
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

        # Save end-of-epoch checkpoint (allows clean resume from the start of the next epoch)
        if fabric.is_global_zero:
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
