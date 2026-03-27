import torch
def run_full_validation(
    fabric,
    model,
    val_loader,
    criterion,
    event_names,
    device,
    event_loss_weights=None,
    tail_weight_scale=1.0,
):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_sq = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sum_smape = torch.zeros(len(event_names), device=device)
    val_sample_count = 0
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
            if "host_dist" in batch:
                batch["host_dist"] = batch["host_dist"].to(device)
                batch["para_dist"] = batch["para_dist"].to(device)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
                host_dist=batch.get("host_dist"),
                para_dist=batch.get("para_dist"),
            )

            target_weights = 1.0 + tail_weight_scale * batch["labels"]
            loss = sum(
                event_loss_weights[i] * (criterion(outputs[:, i], batch["labels"][:, i]) * target_weights[:, i]).mean()
                for i in range(len(event_names))
            )

            preds = outputs
            labels = batch["labels"]

            val_loss += loss.item()
            val_batches += 1

            abs_err = (preds - labels).abs()
            labels_abs = labels.abs()
            sq_err  = abs_err ** 2

            rel_err = abs_err / (labels_abs + 1e-8)
            smape = 2 * abs_err / (preds.abs() + labels_abs + 1e-8)

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

    model.train()

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
            batch["sim_time"] = batch["sim_time"].to(device)  

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch['sim_time'] 
            )

            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if len(preds_list) == 0:
        return None, None
    return torch.cat(preds_list), torch.cat(labels_list)
