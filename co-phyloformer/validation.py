import torch
import torch.nn as nn

def run_full_validation(
    fabric,
    model,
    val_loader,
    criterion,
    event_names,
    device,
    event_loss_weights=None,
    label_mean=None,
    label_std=None,
):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_sq = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sum_smape = torch.zeros(len(event_names), device=device)
    running_min_nonzero = torch.full((len(event_names),), float('inf'), device=device)
    val_sample_count = 0
    all_preds_orig = []  # for histogram logging

    if event_loss_weights is None:
        event_loss_weights = torch.ones(len(event_names), device=device)
    else:
        event_loss_weights = event_loss_weights.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch["host_msa"] = batch["host_msa"].to(device)
            batch["parasite_msa"] = batch["parasite_msa"].to(device)
            batch["sim_time"] = batch["sim_time"].to(device)
            batch["labels"] = batch["labels"].to(device)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
            )

            # Loss in normalized space (same space as training loss).
            if label_mean is not None and label_std is not None:
                labels_norm = (batch["labels"] - label_mean) / label_std
                loss_cosp = criterion(outputs[:, 0], labels_norm[:, 0]).mean()
                loss_sw = criterion(outputs[:, 1], labels_norm[:, 1]).mean()
            else:
                loss_cosp = criterion(outputs[:, 0], batch["labels"][:, 0]).mean()
                loss_sw = criterion(outputs[:, 1], batch["labels"][:, 1]).mean()

            loss = event_loss_weights[0] * loss_cosp + event_loss_weights[1] * loss_sw
            val_loss += loss.item()
            val_batches += 1

            # Denormalize predictions for interpretable metrics.
            if label_mean is not None and label_std is not None:
                preds = outputs * label_std + label_mean
            else:
                preds = outputs

            labels = batch["labels"]
            all_preds_orig.append(preds.cpu())

            abs_err = (preds - labels).abs()
            labels_abs = labels.abs()
            sq_err  = abs_err ** 2

            safe_labels = torch.where(labels_abs > 0, labels_abs, torch.full_like(labels_abs, float('inf')))
            batch_min = torch.amin(safe_labels, dim=0)
            running_min_nonzero = torch.minimum(running_min_nonzero, batch_min)

            fallback_eps = torch.finfo(labels.dtype).eps
            eps_vec = torch.where(torch.isfinite(running_min_nonzero), running_min_nonzero * 1e-2, torch.full_like(running_min_nonzero, fallback_eps))

            rel_err = abs_err / (labels_abs + eps_vec)
            smape = 2 * abs_err / (preds.abs() + labels.abs() + 1e-8)

            val_sum_abs += abs_err.sum(dim=0)
            val_sum_sq  += sq_err.sum(dim=0)
            val_sum_rel += rel_err.sum(dim=0)
            val_sum_smape += smape.sum(dim=0)
            val_sample_count += labels.shape[0]


    # DDP-safe reduction for val_loss: sum losses and sum batches across ranks
    loss_sum_tensor = torch.tensor(val_loss, device=device)
    batches_tensor = torch.tensor(val_batches, device=device)
    loss_sum_tensor = fabric.all_reduce(loss_sum_tensor, reduce_op="sum")
    batches_tensor = fabric.all_reduce(batches_tensor, reduce_op="sum")
    val_loss = (loss_sum_tensor / torch.clamp(batches_tensor, min=1)).item()

    val_sum_abs = fabric.all_reduce(val_sum_abs, reduce_op="sum")
    val_sum_sq  = fabric.all_reduce(val_sum_sq, reduce_op="sum")
    val_sum_rel = fabric.all_reduce(val_sum_rel, reduce_op="sum")
    val_sum_smape = fabric.all_reduce(val_sum_smape, reduce_op="sum")

    val_sample_count = int(fabric.all_reduce(
        torch.tensor(val_sample_count, device=device),
        reduce_op="sum"
    ).item())

    denom = max(1, val_sample_count)
    val_mae = (val_sum_abs / denom).cpu().tolist()
    val_mse = (val_sum_sq / denom).cpu().tolist()
    val_mre = (val_sum_rel / denom).cpu().tolist()
    val_smape = (val_sum_smape / denom).cpu().tolist()

    # Concatenate predictions from this rank for histogram logging.
    val_preds_orig = torch.cat(all_preds_orig, dim=0) if all_preds_orig else None

    return {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_mre": val_mre,
        "val_smape": val_smape,
        "val_preds_orig": val_preds_orig,  # [N, num_events] in original scale
    }


def compute_val_predictions(model, val_loader, device, label_mean=None, label_std=None):
    preds_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch["host_msa"] = batch["host_msa"].to(device)
            batch["parasite_msa"] = batch["parasite_msa"].to(device)
            batch["labels"] = batch["labels"].to(device)
            batch["sim_time"] = batch["sim_time"].to(device)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
            )

            # Denormalize to original scale for plotting.
            if label_mean is not None and label_std is not None:
                outputs = outputs * label_std + label_mean

            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if len(preds_list) == 0:
        return None, None
    return torch.cat(preds_list), torch.cat(labels_list)
