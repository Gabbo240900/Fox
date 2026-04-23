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
    collect_preds=False,
):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sample_count = 0
    if event_loss_weights is None:
        event_loss_weights = torch.ones(len(event_names), device=device)
    else:
        event_loss_weights = event_loss_weights.to(device=device, dtype=torch.float32)

    preds_list  = [] if collect_preds else None
    labels_list = [] if collect_preds else None

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch["host_msa"] = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["sim_time"] = batch["sim_time"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            if "host_dist" in batch:
                batch["host_dist"] = batch["host_dist"].to(device, non_blocking=True)
                batch["para_dist"] = batch["para_dist"].to(device, non_blocking=True)

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

            val_loss += loss.item()
            val_batches += 1
            val_sum_abs += (outputs - batch["labels"]).abs().sum(dim=0)
            val_sum_rel += ((outputs - batch["labels"]).abs() / batch["labels"].clamp(min=0.01)).sum(dim=0)
            val_sample_count += batch["labels"].shape[0]

            if collect_preds and fabric.is_global_zero:
                preds_list.append(outputs.cpu())
                labels_list.append(batch["labels"].cpu())

    loss_sum_tensor = fabric.all_reduce(torch.tensor(val_loss, device=device), reduce_op="sum")
    batches_tensor  = fabric.all_reduce(torch.tensor(val_batches, device=device), reduce_op="sum")
    val_loss = (loss_sum_tensor / torch.clamp(batches_tensor, min=1)).item()

    val_sum_abs = fabric.all_reduce(val_sum_abs, reduce_op="sum")
    val_sum_rel = fabric.all_reduce(val_sum_rel, reduce_op="sum")
    val_sample_count = int(fabric.all_reduce(
        torch.tensor(val_sample_count, device=device), reduce_op="sum"
    ).item())

    denom   = max(1, val_sample_count)
    val_mae = (val_sum_abs / denom).cpu().tolist()
    val_mre = (val_sum_rel / denom).cpu().tolist()

    model.train()

    result = {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mre": val_mre,
        "val_sample_count": val_sample_count,
    }
    if collect_preds and fabric.is_global_zero and preds_list:
        result["val_preds"]  = torch.cat(preds_list)
        result["val_labels"] = torch.cat(labels_list)
    return result


def compute_val_predictions(model, val_loader, device):
    preds_list  = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch["host_msa"]     = batch["host_msa"].to(device, non_blocking=True)
            batch["parasite_msa"] = batch["parasite_msa"].to(device, non_blocking=True)
            batch["labels"]       = batch["labels"].to(device, non_blocking=True)
            batch["sim_time"]     = batch["sim_time"].to(device, non_blocking=True)
            if "host_dist" in batch:
                batch["host_dist"] = batch["host_dist"].to(device, non_blocking=True)
                batch["para_dist"] = batch["para_dist"].to(device, non_blocking=True)

            outputs = model(
                batch["host_msa"],
                batch["parasite_msa"],
                batch["mappings"],
                batch["sim_time"],
                host_dist=batch.get("host_dist"),
                para_dist=batch.get("para_dist"),
            )
            preds_list.append(outputs.cpu())
            labels_list.append(batch["labels"].cpu())

    if not preds_list:
        return None, None
    return torch.cat(preds_list), torch.cat(labels_list)
