import torch
# BEST CONFIGURATION SO FAR FOR SMALL DATASETS
def run_full_validation(fabric, model, val_loader, criterion, event_names, device):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_sq = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sum_smape = torch.zeros(len(event_names), device=device)
    running_min_nonzero = torch.full((len(event_names),), float('inf'), device=device)
    val_sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch["host_msa"] = batch["host_msa"].to(device)
            batch["parasite_msa"] = batch["parasite_msa"].to(device)
            batch["sim_time"] = batch["sim_time"].to(device)  # --- sim_time disabled ---
            batch["labels"] = batch["labels"].to(device)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch['sim_time']  # --- sim_time disabled ---
            )

            loss_cosp = criterion(outputs[:, 0], batch["labels"][:, 0]).mean()
            loss_sw   = criterion(outputs[:, 1], batch["labels"][:, 1]).mean()
            loss = loss_cosp + loss_sw

            preds = outputs
            labels = batch["labels"]

            val_loss += loss.item()
            val_batches += 1

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

    val_loss_tensor = torch.tensor(val_loss / max(1, val_batches), device=device)
    val_loss = fabric.all_reduce(val_loss_tensor, reduce_op="mean").item()

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

    return {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_mre": val_mre,
        "val_smape": val_smape,
    }


def compute_val_predictions(model, val_loader, device):
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
            batch["sim_time"] = batch["sim_time"].to(device)  # --- sim_time disabled ---

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                None,  # --- sim_time disabled ---
            )

            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if len(preds_list) == 0:
        return None, None
    return torch.cat(preds_list), torch.cat(labels_list)
