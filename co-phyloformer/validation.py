import torch

# Quantiles must match model.py / train.py
QUANTILES = (0.5, 0.75, 0.9)
Q50_INDEX = QUANTILES.index(0.5)
Q90_INDEX = QUANTILES.index(0.9)


def _pinball_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """Pinball loss for quantile regression.

    pred_q: (B, Q) predicted quantiles in [0,1]
    y:      (B,)   target in [0,1]
    returns: scalar tensor
    """
    qs = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype).view(1, -1)  # (1, Q)
    yq = y.view(-1, 1)  # (B, 1)
    diff = yq - pred_q  # (B, Q)
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)
    return loss.mean()


def run_full_validation(fabric, model, val_loader, criterion, event_names, device):
    """Full validation.

    - model outputs: (B, 2Q) where Q=len(QUANTILES)
      order: [cospec_q1..qQ, switch_q1..qQ]
    - loss: pinball loss on all quantiles
    - metrics: computed on q50 point estimate only

    `criterion` is ignored (kept for backward compatibility with older training code).
    """
    model.eval()

    val_loss_sum = 0.0
    val_batches = 0

    val_sum_abs = torch.zeros(len(event_names), device=device)
    val_sum_sq = torch.zeros(len(event_names), device=device)
    val_sum_rel = torch.zeros(len(event_names), device=device)
    val_sum_smape = torch.zeros(len(event_names), device=device)
    running_min_nonzero = torch.full((len(event_names),), float('inf'), device=device)
    val_sample_count = 0

    # Optional calibration: coverage of q90
    cov90_sum = torch.zeros(len(event_names), device=device)

    base_model = getattr(model, "module", model)

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

            # Split quantiles: each (B, Q)
            cos_q, sw_q = base_model.split_quantiles(outputs)

            # Pinball loss on all quantiles
            loss_cosp = _pinball_loss(cos_q, batch["labels"][:, 0], quantiles=QUANTILES)
            loss_sw   = _pinball_loss(sw_q,  batch["labels"][:, 1], quantiles=QUANTILES)
            loss = loss_cosp + loss_sw

            val_loss_sum += float(loss.item())
            val_batches += 1

            # Metrics in probability space using q50 only
            preds = torch.stack([cos_q[:, Q50_INDEX], sw_q[:, Q50_INDEX]], dim=1)  # (B, 2)
            labels = batch["labels"]

            abs_err = (preds - labels).abs()
            labels_abs = labels.abs()
            sq_err = abs_err ** 2

            safe_labels = torch.where(labels_abs > 0, labels_abs, torch.full_like(labels_abs, float('inf')))
            batch_min = torch.amin(safe_labels, dim=0)
            running_min_nonzero = torch.minimum(running_min_nonzero, batch_min)

            fallback_eps = torch.finfo(labels.dtype).eps
            eps_vec = torch.where(
                torch.isfinite(running_min_nonzero),
                running_min_nonzero * 1e-2,
                torch.full_like(running_min_nonzero, fallback_eps),
            )

            rel_err = abs_err / (labels_abs + eps_vec)
            smape = 2 * abs_err / (preds.abs() + labels_abs + eps_vec)

            val_sum_abs += abs_err.sum(dim=0)
            val_sum_sq += sq_err.sum(dim=0)
            val_sum_rel += rel_err.sum(dim=0)
            val_sum_smape += smape.sum(dim=0)
            val_sample_count += labels.shape[0]

            # Coverage of q90: P(y <= q90_pred) should be ~0.90 if calibrated
            preds_q90 = torch.stack([cos_q[:, Q90_INDEX], sw_q[:, Q90_INDEX]], dim=1)
            cov90 = (labels <= preds_q90).to(dtype=torch.float32).mean(dim=0)  # (2,)
            cov90_sum += cov90

    # DDP-safe reduction for val_loss: sum losses and sum batches across ranks
    loss_sum_tensor = torch.tensor(val_loss_sum, device=device)
    batches_tensor = torch.tensor(val_batches, device=device)
    loss_sum_tensor = fabric.all_reduce(loss_sum_tensor, reduce_op="sum")
    batches_tensor = fabric.all_reduce(batches_tensor, reduce_op="sum")
    val_loss = (loss_sum_tensor / torch.clamp(batches_tensor, min=1)).item()

    # Reduce metric accumulators
    val_sum_abs = fabric.all_reduce(val_sum_abs, reduce_op="sum")
    val_sum_sq = fabric.all_reduce(val_sum_sq, reduce_op="sum")
    val_sum_rel = fabric.all_reduce(val_sum_rel, reduce_op="sum")
    val_sum_smape = fabric.all_reduce(val_sum_smape, reduce_op="sum")

    val_sample_count = int(
        fabric.all_reduce(torch.tensor(val_sample_count, device=device), reduce_op="sum").item()
    )

    cov90_sum = fabric.all_reduce(cov90_sum, reduce_op="sum")
    # Average coverage across batches (not samples). Good enough as a diagnostic.
    cov90 = (cov90_sum / torch.clamp(batches_tensor, min=1)).detach().cpu().tolist()

    denom = max(1, val_sample_count)
    val_mae = (val_sum_abs / denom).detach().cpu().tolist()
    val_mse = (val_sum_sq / denom).detach().cpu().tolist()
    val_mre = (val_sum_rel / denom).detach().cpu().tolist()
    val_smape = (val_sum_smape / denom).detach().cpu().tolist()

    return {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_mre": val_mre,
        "val_smape": val_smape,
        "val_cov90": cov90,
    }


def compute_val_predictions(model, val_loader, device):
    """Return raw model outputs and labels.

    For quantile regression, outputs are (B, 2Q).
    Downstream code (train.py) reduces to q50 for plots/CSV.
    """
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

            preds_list.append(outputs.detach().cpu())
            labels_list.append(batch["labels"].detach().cpu())

    if len(preds_list) == 0:
        return None, None

    return torch.cat(preds_list, dim=0), torch.cat(labels_list, dim=0)
