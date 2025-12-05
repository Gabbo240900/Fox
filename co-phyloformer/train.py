
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import Cophyloformer
from data import CophylogenyDataset
from torch.nn import functional as F
import time
import numpy as np
import os
from plot import plot_event_metric_over_epochs, plot_epoch_loss_curve, plot_labels_vs_predictions
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
from itertools import islice

import wandb
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric.strategies import DDPStrategy
from validation import run_full_validation, compute_val_predictions

import glob


torch.set_float32_matmul_precision('high')

seed_everything(42)
event_names = [
    "Cospeciations",
    "Host_spread/Switches"
]

start_time = time.time()  # Record start time

class LazyCophyloformerDataset(Dataset):
    def __init__(self, preencoded_dir):
        self.pt_files = sorted(glob.glob(os.path.join(preencoded_dir, "*.pt")))
        self.preencoded_dir = preencoded_dir

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Skip empty samples
        if len(sample["host_msas"]) == 0 or len(sample["parasite_msas"]) == 0:
            return None

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

        sim_time = torch.tensor([sample["event_frequencies"].get("Sim_time", 0.0)], dtype=torch.float32)
        return {
            "host_msa": torch.stack([
                (encode_sequence(seq))
                for seq in sample["host_msas"].values()
            ]),
            "parasite_msa": torch.stack([
                (encode_sequence(seq))
                for seq in sample["parasite_msas"].values()
            ]),
            "mappings": valid_mappings,
            "labels": labels,
            "sim_time": sim_time,
        }
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, filename, batch_idx=None):
    """Save model and optimizer state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    if batch_idx is not None:
        checkpoint['batch_idx'] = batch_idx
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    """Load model and optimizer state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    batch_idx = checkpoint.get('batch_idx', None)
    return epoch, val_loss, batch_idx

def collate_fn(batch):
    """Pads MSA sequences dynamically to match batch size."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    host_msas = [torch.cat([torch.full((1, sample["host_msa"].shape[1]), 22), sample["host_msa"]], dim=0) for sample in batch]
    parasite_msas = [torch.cat([torch.full((1, sample["parasite_msa"].shape[1]), 22), sample["parasite_msa"]], dim=0) for sample in batch]
    labels = torch.stack([sample["labels"] for sample in batch])
    mappings = [sample["mappings"] for sample in batch]  
    sim_time = torch.stack([sample["sim_time"] for sample in batch])
    sim_time_min = sim_time.min()
    sim_time_max = sim_time.max()
    sim_time = (sim_time - sim_time_min) / (sim_time_max - sim_time_min + 1e-8)

    #  Fix: Ensure consistent padding for batch processing 
    max_host_len = max(m.shape[0] for m in host_msas)
    max_parasite_len = max(m.shape[0] for m in parasite_msas)

    host_msas = [F.pad(m, (0, 0, 0, max_host_len - m.shape[0]), value=0) for m in host_msas]
    parasite_msas = [F.pad(m, (0, 0, 0, max_parasite_len - m.shape[0]), value=0) for m in parasite_msas]

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
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"  # Standard amino acids + gap
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    padding_token = 22  # Ensure padding has a consistent index

    encoded = [aa_to_index.get(aa, padding_token) for aa in sequence[:max_len]]

    # Pad to max length
    encoded += [padding_token] * (max_len - len(encoded))

    return torch.tensor(encoded, dtype=torch.long)

def main(fabric: Fabric):
    # Load Data
    preencoded_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/preencoded_pt/"
    #preencoded_dir = os.path.join(os.environ["JOBSCRATCH"], "preencoded_pt")
    #preencoded_dir = '/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/test/'
    #dataset_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Dataset_final/"
    #dataset_dir = "../generate_treeducken/generated_trees/Datasets/"
    #dataset = CophylogenyDataset(dataset_dir).get_data()
    dataset = LazyCophyloformerDataset(preencoded_dir)
    # Train/Validation Split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset   = torch.utils.data.Subset(dataset, val_indices)
    device = fabric.device
    epochs = 5

    batch_size = 40

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=False
    )
    # Create validation loader
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False
    )
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    lr = 1e-4 # lower learning rate (5e-5, or 1e-5).
    wd = 0
    #criterion = nn.HuberLoss(reduction='none', delta=1.0)
    criterion = nn.L1Loss(reduction='none')# Trying optimizing MAE instead of huber
    # criterion = nn.MSELoss(reduction='none')

    model = Cophyloformer()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model, optimizer = fabric.setup(model, optimizer)

    # Resume from checkpoint logic
    checkpoint_dir = "checkpoints"
    resume_ckpt_path = os.environ.get("RESUME_CKPT", None)
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        if fabric.is_global_zero:
            print(f"[Checkpoint] Loading checkpoint from {resume_ckpt_path}")
        loaded_epoch, loaded_val_loss, loaded_batch_idx = load_checkpoint(model, optimizer, resume_ckpt_path, map_location=fabric.device)
        # If batch_idx is present, resume from that batch in the epoch
        start_epoch = loaded_epoch
        start_batch = (loaded_batch_idx + 1) if loaded_batch_idx is not None else 0
        best_val_loss = loaded_val_loss
        if fabric.is_global_zero:
            print(f"[Checkpoint] Resumed from epoch {loaded_epoch}, best_val_loss {loaded_val_loss}, batch_idx {loaded_batch_idx}")

    total_steps = epochs * len(train_loader)
    warmup_steps = total_steps // 10 # 10% warmup steps 
    #warmup_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
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
                "dataset_dir": preencoded_dir,
                "scheduler": "linear_warmup",
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "model_name": model.__class__.__name__,
                "dataset_size": len(dataset),  
                **model_config                 
            },
        )

        # Log total learnable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run.config["num_parameters"] = num_params

        # Watch gradients and parameters
        wandb.watch(getattr(model, "module", model), log="all", log_freq=100)


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

    val_predictions_data = []  
    best_step_predictions = None

    # Training loop over all batches per epoch (no micro-epochs)
    for epoch in range(start_epoch, epochs):
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
            batch["host_msa"] = batch["host_msa"].to(device)
            batch["parasite_msa"] = batch["parasite_msa"].to(device)
            batch["sim_time"] = batch["sim_time"].to(device)
            batch["labels"] = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
            )

            for idx in range(outputs.shape[0]):
                all_train_prediction_data.append({
                    "Sample_Index": batch_idx * outputs.shape[0] + idx,
                    "Cospeciations_Pred": outputs[idx, 0].item(),
                    "Cospeciations_GT": batch["labels"][idx, 0].item(),
                    "Host_switches_Pred": outputs[idx, 1].item(),
                    "Host_switches_GT": batch["labels"][idx, 1].item(),
                })

            loss_cospeciation = criterion(outputs[:, 0], batch["labels"][:, 0]).mean()
            loss_switches     = criterion(outputs[:, 1], batch["labels"][:, 1]).mean()
            total_loss_tensor = loss_cospeciation + loss_switches

            fabric.backward(total_loss_tensor)
            optimizer.step()
            lr_scheduler.step()

            #Log training loss every 100 batches 
            if fabric.is_global_zero and ((batch_idx + 1) % 1200 == 0 or batch_idx == 0):
                val_results = run_full_validation(fabric, model, val_loader, criterion, event_names, device)
                wandb.log({
                    "train/loss_step": total_loss_tensor.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "step": epoch * len(train_loader) + batch_idx,
                    "val/loss_step": val_results["val_loss"],
                    **{f"val/MAE_step/{event_names[i]}": val_results["val_mae"][i] for i in range(len(event_names))},
                    **{f"val/MSE_step/{event_names[i]}": val_results["val_mse"][i] for i in range(len(event_names))},
                    **{f"val/MRE_step/{event_names[i]}": val_results["val_mre"][i] for i in range(len(event_names))},
                    **{f"val/sMAPE_step/{event_names[i]}": val_results["val_smape"][i] for i in range(len(event_names))}
                })
            checkpoints_per_epoch = 5
            save_every = max(1, len(train_loader) // checkpoints_per_epoch)
            if fabric.is_global_zero and (batch_idx + 1) % save_every == 0:
                mid_ckpt_name = f"epoch{epoch+1}_batch{batch_idx+1}.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    total_loss_tensor.item(),
                    checkpoint_dir,
                    mid_ckpt_name,
                    batch_idx=batch_idx
                )
                print(f"[Checkpoint] Saved mid-epoch checkpoint: {mid_ckpt_name}")

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

                # --- Log per-event metrics every 100 batches ---
                if fabric.is_global_zero and ((batch_idx + 1) % 1200 == 0 or batch_idx == 0):
                    batch_mae   = abs_err.mean(dim=0).detach().cpu().tolist()
                    batch_mse   = sq_err.mean(dim=0).detach().cpu().tolist()
                    batch_mre   = rel_err.mean(dim=0).detach().cpu().tolist()
                    batch_smape = smape.mean(dim=0).detach().cpu().tolist()

                    log_dict = {
                        "step": epoch * len(train_loader) + batch_idx,
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

        # All-reduce epoch loss across all processes so that the logged value
        # represents the global average and not just rank 0.
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        epoch_loss = fabric.all_reduce(epoch_loss_tensor, reduce_op="mean").item()

        epoch_losses.append(epoch_loss)

        if fabric.is_global_zero:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']}")

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
                        pred_val = outputs[sample_idx, i].item()
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
        val_results = run_full_validation(fabric, model, val_loader, criterion, event_names, device)
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

        model.train()

        # --- Save best-step checkpoint safely 
        if fabric.is_global_zero and hasattr(main, "best_step_batch"):
            # Remove previous best-step checkpoint if it exists
            if hasattr(main, "best_step_ckpt_name") and main.best_step_ckpt_name is not None:
                old_path = os.path.join(checkpoint_dir, main.best_step_ckpt_name)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"[Checkpoint] Removed old BEST STEP checkpoint: {main.best_step_ckpt_name}")

            # Save new best-step checkpoint
            step_ckpt_name = (
                f"best_step_val_loss_epoch{main.best_step_epoch+1}_batch{main.best_step_batch+1}.pth"
            )

            save_checkpoint(
                model,
                optimizer,
                main.best_step_epoch,
                main.best_step_val_loss,
                checkpoint_dir,
                step_ckpt_name,
                batch_idx=main.best_step_batch
            )

            main.best_step_ckpt_name = step_ckpt_name
            print(f"[Checkpoint] Saved NEW BEST STEP validation model (SAFE): {step_ckpt_name}")

            # Clear temporary tracking so it doesn't trigger again this epoch
            del main.best_step_batch
            del main.best_step_epoch

        #  Checkpoint saving logic at end of epoch 
        if fabric.is_global_zero:
            # Save checkpoint for current epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_loss,
                checkpoint_dir,
                f"epoch_{epoch+1}.pth"
            )
            # Save best (lowest validation loss) checkpoint
            if val_loss < best_val_loss:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    checkpoint_dir,
                    "best_val_loss.pth"
                )
                best_val_loss = val_loss

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

        # Log generated plots as images to W&B
        wandb.log({
            "plots/loss_curve": wandb.Image("combined_loss.png"),
        })
        for metric_name in ["MAE", "MSE", "MRE", "sMAPE"]:
            png_name = f"{metric_name.lower()}_over_epochs.png"
            if os.path.exists(png_name):
                wandb.log({f"plots/{metric_name}": wandb.Image(png_name)})

        # Save final epoch predictions to a CSV file
        import csv

        output_path = "final_predictions.csv"
        with open(output_path, mode="w", newline="") as csv_file:
            fieldnames = ["Sample_Index", "Cospeciations_Pred", "Cospeciations_GT", "Host_switches_Pred", "Host_switches_GT"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            if hasattr(main, "best_step_predictions") and main.best_step_predictions is not None:
                writer.writerows(main.best_step_predictions)
            else:
                writer.writerows(val_predictions_data)

        print(f"Final predictions saved to {output_path}")


        # --- Combined Train + Val Scatter Plots (Pred vs GT) ---
        val_preds_tensor, val_labels_tensor = compute_val_predictions(model, val_loader, device)

        if val_preds_tensor is not None:
            # Cospeciations
            plot_labels_vs_predictions(
                train_labels=[row["Cospeciations_GT"] for row in all_train_prediction_data],
                train_preds=[row["Cospeciations_Pred"] for row in all_train_prediction_data],
                val_labels=val_labels_tensor[:, 0].numpy(),
                val_preds=val_preds_tensor[:, 0].numpy(),
                event_name="Cospeciations",
                filename="combined_label_vs_pred_cospeciations.png"
            )

            # Host switches
            plot_labels_vs_predictions(
                train_labels=[row["Host_switches_GT"] for row in all_train_prediction_data],
                train_preds=[row["Host_switches_Pred"] for row in all_train_prediction_data],
                val_labels=val_labels_tensor[:, 1].numpy(),
                val_preds=val_preds_tensor[:, 1].numpy(),
                event_name="Host Switches",
                filename="combined_label_vs_pred_switches.png"
            )
        else:
            print("[Warning] No validation predictions available for scatter plots.")
        # Log scatter plots to W&B
        if os.path.exists("label_vs_pred_cospeciations.png"):
            wandb.log({"plots/labels_vs_preds_cospeciations": wandb.Image("label_vs_pred_cospeciations.png")})
        if os.path.exists("label_vs_pred_host_switches.png"):
            wandb.log({"plots/labels_vs_preds_host_switches": wandb.Image("label_vs_pred_host_switches.png")})

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
        strategy=DDPStrategy(
                find_unused_parameters=True,
            )
    )
    fabric.launch(main)
