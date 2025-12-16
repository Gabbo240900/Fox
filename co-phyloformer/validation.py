import torch
# BEST CONFIGURATION SO FAR FOR SMALL DATASETS
def run_full_validation(
    fabric,
    model,
    val_loader,
    criterion,
    event_names,
    device,
    cospec_bin_edges=None,
):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_sq = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sum_smape = torch.zeros(len(event_names), device=device)
    val_sample_count = 0

    # Bin-wise MAE accumulators (Cospeciations only)
    if cospec_bin_edges is not None:
        num_bins = len(cospec_bin_edges) - 1
        bin_abs_sum = torch.zeros(num_bins, device=device)
        bin_count   = torch.zeros(num_bins, device=device)

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
                batch["sim_time"],
            )

            loss_cosp = criterion(outputs[:, 0], batch["labels"][:, 0]).mean()
            loss_sw   = criterion(outputs[:, 1], batch["labels"][:, 1]).mean()
            loss = loss_cosp + loss_sw

            preds = outputs
            labels = batch["labels"]

            val_loss += loss.item()
            val_batches += 1

            abs_err = (preds - labels).abs()

            # --- Bin-wise accumulation for Cospeciations ---
            if cospec_bin_edges is not None:
                cospec_gt = labels[:, 0]
                cospec_pred = preds[:, 0]

                bin_ids = torch.bucketize(
                    cospec_gt,
                    cospec_bin_edges[1:-1].to(device),
                    right=True
                )

                for b in range(num_bins):
                    mask = bin_ids == b
                    if mask.any():
                        bin_abs_sum[b] += (cospec_pred[mask] - cospec_gt[mask]).abs().sum()
                        bin_count[b]   += mask.sum()

            sq_err  = abs_err ** 2

            denom_abs = torch.where(labels.abs() > 0, labels.abs(), torch.full_like(labels, 1e9))
            rel_err = abs_err / denom_abs
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

    if cospec_bin_edges is not None:
        bin_abs_sum = fabric.all_reduce(bin_abs_sum, reduce_op="sum")
        bin_count   = fabric.all_reduce(bin_count, reduce_op="sum")

    denom = max(1, val_sample_count)
    val_mae = (val_sum_abs / denom).cpu().tolist()
    val_mse = (val_sum_sq / denom).cpu().tolist()
    val_mre = (val_sum_rel / denom).cpu().tolist()
    val_smape = (val_sum_smape / denom).cpu().tolist()

    val_bin_mae = None
    if cospec_bin_edges is not None:
        val_bin_mae = {}
        for b in range(num_bins):
            if bin_count[b] > 0:
                val_bin_mae[f"bin_{b}"] = (bin_abs_sum[b] / bin_count[b]).item()
            else:
                val_bin_mae[f"bin_{b}"] = float("nan")

    return {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_mre": val_mre,
        "val_smape": val_smape,
        "val_bin_mae": val_bin_mae,
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
            batch["sim_time"] = batch["sim_time"].to(device)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
            )

            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if len(preds_list) == 0:
        return None, None
    return torch.cat(preds_list), torch.cat(labels_list)
