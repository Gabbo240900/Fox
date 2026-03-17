"""
evaluate.py — Evaluation script for Co-Phyloformer.

Predicts 4 evolutionary event frequencies:
  [0] Speciation_freq
  [1] HGT_freq
  [2] Loss_freq
  [3] Duplication_freq

Usage examples:
  python evaluate.py --checkpoint ./checkpoints/best_model.pt --data_dir ../generate_asymmetree/generated_trees/Datasets
  python evaluate.py --sanity_check --data_dir ../generate_asymmetree/generated_trees/Datasets
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Optional matplotlib — handle missing gracefully
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving files
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "matplotlib is not installed. Plotting will be disabled.", ImportWarning
    )

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]
EVENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Small epsilon to avoid log(0) in KL/JS divergence
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics between predictions and targets.

    Args:
        preds:   [N, 4] predicted frequencies (float)
        targets: [N, 4] true frequencies (float)

    Returns:
        dict with keys:
            mae_per_class       [4]  — MAE for each event type
            rmse_per_class      [4]  — RMSE for each event type
            mae_overall         scalar — mean MAE across all classes
            pearson_per_class   [4]  — Pearson r per event
            spearman_per_class  [4]  — Spearman rho per event
            kl_divergence       scalar — mean KL(target || pred)
            js_divergence       scalar — mean Jensen-Shannon divergence
            speciation_vs_hgt_mae  scalar — MAE for spec + HGT columns only
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    assert preds.shape == targets.shape, "Shape mismatch between preds and targets"
    assert preds.ndim == 2 and preds.shape[1] == 4, "Expected shape [N, 4]"

    n = preds.shape[0]
    errors = np.abs(preds - targets)

    # Per-class MAE and RMSE
    mae_per_class = errors.mean(axis=0)
    rmse_per_class = np.sqrt((errors ** 2).mean(axis=0))

    # Overall MAE (mean over all classes and samples)
    mae_overall = float(errors.mean())

    # Pearson and Spearman per class
    pearson_per_class = np.zeros(4)
    spearman_per_class = np.zeros(4)
    for i in range(4):
        if n > 1:
            r, _ = pearsonr(targets[:, i], preds[:, i])
            rho, _ = spearmanr(targets[:, i], preds[:, i])
            pearson_per_class[i] = r
            spearman_per_class[i] = rho
        else:
            pearson_per_class[i] = float("nan")
            spearman_per_class[i] = float("nan")

    # KL divergence: KL(target || pred)  — mean over samples
    p = np.clip(targets, _EPS, None)
    q = np.clip(preds, _EPS, None)
    # Normalise rows so they are valid distributions (guard against numerical drift)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    kl_per_sample = (p * np.log(p / q)).sum(axis=1)
    kl_divergence = float(kl_per_sample.mean())

    # Jensen-Shannon divergence (symmetric, bounded [0,1] for log base 2)
    m = 0.5 * (p + q)
    js_per_sample = 0.5 * (p * np.log(p / m)).sum(axis=1) + \
                    0.5 * (q * np.log(q / m)).sum(axis=1)
    js_divergence = float(js_per_sample.mean())

    # Key biological metric: MAE for Speciation (0) and HGT (1) only
    speciation_vs_hgt_mae = float(errors[:, :2].mean())

    return {
        "mae_per_class":        mae_per_class.tolist(),
        "rmse_per_class":       rmse_per_class.tolist(),
        "mae_overall":          mae_overall,
        "pearson_per_class":    pearson_per_class.tolist(),
        "spearman_per_class":   spearman_per_class.tolist(),
        "kl_divergence":        kl_divergence,
        "js_divergence":        js_divergence,
        "speciation_vs_hgt_mae": speciation_vs_hgt_mae,
    }


# ---------------------------------------------------------------------------
# Model evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(model, data_loader, device) -> tuple:
    """
    Run inference on all samples in data_loader.

    Returns:
        predictions_np: np.ndarray [N, 4]
        targets_np:     np.ndarray [N, 4]
        metrics_dict:   dict from compute_metrics()
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            host_seqs     = batch["host_seqs"]
            parasite_seqs = batch["parasite_seqs"]
            mappings      = batch["mappings"]
            labels        = batch["labels"]  # FloatTensor [batch, 4] or [4]

            # Move tensors to device where applicable
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            pred = model(host_seqs, parasite_seqs, mappings)  # [4] or [batch, 4]

            # Ensure 2-D
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    predictions_np = np.concatenate(all_preds, axis=0)
    targets_np     = np.concatenate(all_targets, axis=0)
    metrics_dict   = compute_metrics(predictions_np, targets_np)

    return predictions_np, targets_np, metrics_dict


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, split_name: str = "test") -> None:
    """Pretty-print metric table to stdout."""
    width = 60
    sep = "=" * width
    print(f"\n{sep}")
    print(f"  Co-Phyloformer Evaluation Results  [{split_name.upper()} split]")
    print(sep)

    # Per-class table
    col_w = 14
    header = f"{'Event':<14}" + "".join(f"{n:>{col_w}}" for n in EVENT_NAMES)
    print(f"\n{header}")
    print("-" * (14 + col_w * 4))

    for metric_key, label in [
        ("mae_per_class",     "MAE"),
        ("rmse_per_class",    "RMSE"),
        ("pearson_per_class", "Pearson r"),
        ("spearman_per_class","Spearman rho"),
    ]:
        vals = metrics[metric_key]
        row = f"{label:<14}" + "".join(f"{v:>{col_w}.4f}" for v in vals)
        print(row)

    print()
    print(f"  {'Overall MAE':<30}: {metrics['mae_overall']:.4f}")
    print(f"  {'KL Divergence':<30}: {metrics['kl_divergence']:.4f}")
    print(f"  {'JS Divergence':<30}: {metrics['js_divergence']:.4f}")
    print(f"  {'Speciation vs HGT MAE':<30}: {metrics['speciation_vs_hgt_mae']:.4f}  [KEY METRIC]")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(preds: np.ndarray, targets: np.ndarray, save_dir: str = "./eval_plots") -> None:
    """
    Create and save evaluation plots.

    Saves to {save_dir}/plots/:
      1. scatter_<event>.png       — predicted vs true per event type
      2. error_hist_<event>.png    — error distribution histograms
      3. correlation_heatmap.png   — heatmap of pred/true column correlations
      4. mae_bar.png               — bar chart of MAE per event
      5. spec_vs_hgt.png           — Speciation vs HGT scatter (true vs pred)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] matplotlib not available — skipping plots.")
        return

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    n = preds.shape[0]
    errors = np.abs(preds - targets)

    # --- 1. Scatter plots: predicted vs true for each event type ---
    for i, (name, color) in enumerate(zip(EVENT_NAMES, EVENT_COLORS)):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(targets[:, i], preds[:, i], c=color, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
        lims = [min(targets[:, i].min(), preds[:, i].min()) - 0.02,
                max(targets[:, i].max(), preds[:, i].max()) + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"True {name}_freq")
        ax.set_ylabel(f"Predicted {name}_freq")
        ax.set_title(f"{name}: Predicted vs True (N={n})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"scatter_{name.lower()}.png"), dpi=150)
        plt.close(fig)

    # --- 2. Error distribution histograms per class ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, (name, color) in enumerate(zip(EVENT_NAMES, EVENT_COLORS)):
        ax = axes[i]
        ax.hist(errors[:, i], bins=30, color=color, edgecolor="k", alpha=0.8)
        ax.axvline(errors[:, i].mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean={errors[:,i].mean():.3f}")
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Error Distribution")
        ax.legend(fontsize=8)
    fig.suptitle("Per-Class Error Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "error_histograms.png"), dpi=150)
    plt.close(fig)

    # --- 3. Correlation heatmap: pred and true columns ---
    # Build combined matrix: [true_spec, true_hgt, ..., pred_spec, pred_hgt, ...]
    combined = np.hstack([targets, preds])
    col_labels = [f"True_{n[:3]}" for n in EVENT_NAMES] + [f"Pred_{n[:3]}" for n in EVENT_NAMES]
    corr_matrix = np.corrcoef(combined.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(col_labels, fontsize=9)
    for r in range(8):
        for c in range(8):
            ax.text(c, r, f"{corr_matrix[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(corr_matrix[r, c]) < 0.7 else "white")
    ax.set_title("Correlation Heatmap: True vs Predicted Frequencies")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- 4. Bar chart: MAE per event type ---
    mae_per_class = errors.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(EVENT_NAMES, mae_per_class, color=EVENT_COLORS, edgecolor="k", alpha=0.85)
    for bar, val in zip(bars, mae_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("MAE per Event Type")
    ax.set_ylim(0, mae_per_class.max() * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mae_bar.png"), dpi=150)
    plt.close(fig)

    # --- 5. Speciation vs HGT scatter (true vs predicted, colored by total error) ---
    total_error = errors.mean(axis=1)  # mean error per sample, used for coloring

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True Spec vs HGT
    sc0 = axes[0].scatter(targets[:, 0], targets[:, 1], c=total_error,
                          cmap="YlOrRd", alpha=0.7, edgecolors="k", linewidths=0.3, s=50)
    plt.colorbar(sc0, ax=axes[0], label="Mean Abs Error")
    axes[0].set_xlabel("True Speciation_freq")
    axes[0].set_ylabel("True HGT_freq")
    axes[0].set_title("True: Speciation vs HGT")

    # Predicted Spec vs HGT
    sc1 = axes[1].scatter(preds[:, 0], preds[:, 1], c=total_error,
                          cmap="YlOrRd", alpha=0.7, edgecolors="k", linewidths=0.3, s=50)
    plt.colorbar(sc1, ax=axes[1], label="Mean Abs Error")
    axes[1].set_xlabel("Pred Speciation_freq")
    axes[1].set_ylabel("Pred HGT_freq")
    axes[1].set_title("Predicted: Speciation vs HGT")

    fig.suptitle("Speciation vs HGT Trade-off (colored by prediction error)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "spec_vs_hgt.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(data_dir: str, num_samples: int = 5) -> None:
    """
    Test that data loading works without a trained model.
    Prints summary statistics for num_samples samples.
    """
    print(f"\n{'='*55}")
    print("  Co-Phyloformer — Data Loading Sanity Check")
    print(f"{'='*55}")
    print(f"  data_dir   : {data_dir}")
    print(f"  num_samples: {num_samples}\n")

    try:
        from dataset import CoPhyloDataset, get_dataloaders
    except ImportError as e:
        print(f"[ERROR] Could not import dataset module: {e}")
        print("        Make sure you run this script from the co_phyloformer/ directory.")
        sys.exit(1)

    try:
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=1)
    except Exception as e:
        print(f"[ERROR] get_dataloaders() failed: {e}")
        sys.exit(1)

    loader_info = [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]

    for split_name, loader in loader_info:
        n_total = len(loader.dataset) if hasattr(loader, "dataset") else "?"
        print(f"  [{split_name:>5}] dataset size: {n_total}")

    print()

    # Inspect first num_samples from the test split
    test_iter = iter(test_loader)
    all_labels = []
    for idx in range(num_samples):
        try:
            batch = next(test_iter)
        except StopIteration:
            print(f"  [INFO] Only {idx} samples available in test split.")
            break

        # collate_fn returns a list of sample dicts
        sample = batch[0] if isinstance(batch, list) else batch

        labels = sample.get("labels")
        meta   = sample.get("metadata", {})

        host_seqs     = sample.get("host_seqs", {})
        parasite_seqs = sample.get("parasite_seqs", {})
        mappings      = sample.get("mappings", [])

        n_host     = len(host_seqs) if isinstance(host_seqs, dict) else "?"
        n_parasite = len(parasite_seqs) if isinstance(parasite_seqs, dict) else "?"
        n_mappings = len(mappings) if hasattr(mappings, "__len__") else "?"

        print(f"  Sample {idx}:")
        print(f"    host sequences    : {n_host}")
        print(f"    parasite sequences: {n_parasite}")
        print(f"    mappings          : {n_mappings}")

        if labels is not None:
            lbl_np = labels.numpy().flatten() if isinstance(labels, torch.Tensor) else np.array(labels).flatten()
            all_labels.append(lbl_np)
            label_str = "  ".join(f"{EVENT_NAMES[i]}={lbl_np[i]:.4f}" for i in range(4))
            print(f"    labels            : [{label_str}]")
            print(f"    label sum         : {lbl_np.sum():.4f}")

        if meta:
            print(f"    metadata keys     : {list(meta.keys())}")
        print()

    if all_labels:
        arr = np.stack(all_labels, axis=0)
        print("  Label statistics across inspected samples:")
        for i, name in enumerate(EVENT_NAMES):
            print(f"    {name:<14}: mean={arr[:,i].mean():.4f}  std={arr[:,i].std():.4f}"
                  f"  min={arr[:,i].min():.4f}  max={arr[:,i].max():.4f}")

    print(f"\n{'='*55}")
    print("  Sanity check complete — data loading works.")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[INFO] Metrics saved to: {path}")


def save_predictions(preds: np.ndarray, targets: np.ndarray, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "predictions.csv")
    with open(path, "w") as fh:
        fh.write("sample_idx,spec_pred,hgt_pred,loss_pred,dup_pred,"
                 "spec_true,hgt_true,loss_true,dup_true\n")
        for i in range(preds.shape[0]):
            row = (f"{i},"
                   f"{preds[i,0]:.6f},{preds[i,1]:.6f},{preds[i,2]:.6f},{preds[i,3]:.6f},"
                   f"{targets[i,0]:.6f},{targets[i,1]:.6f},{targets[i,2]:.6f},{targets[i,3]:.6f}")
            fh.write(row + "\n")
    print(f"[INFO] Predictions saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Co-Phyloformer checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file (required unless --sanity_check).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Datasets/ folder.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which data split(s) to evaluate.")
    parser.add_argument("--save_dir", type=str, default="./eval_results",
                        help="Directory to save results.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable matplotlib plotting.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on: 'auto', 'cpu', 'cuda', 'mps'.")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Sanity check mode
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run data-loading sanity check without a trained model.")
    parser.add_argument("--sanity_num_samples", type=int, default=5,
                        help="Number of samples to inspect during sanity check.")

    return parser


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Sanity check mode ---
    if args.sanity_check:
        sanity_check(args.data_dir, num_samples=args.sanity_num_samples)
        return

    # --- Full evaluation requires a checkpoint ---
    if args.checkpoint is None:
        parser.error("--checkpoint is required for full evaluation (or use --sanity_check).")

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Imports that require the package
    try:
        from model import CoPhyloformer
        from dataset import get_dataloaders
    except ImportError as e:
        print(f"[ERROR] Could not import model/dataset: {e}")
        print("        Run from the co_phyloformer/ directory.")
        sys.exit(1)

    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Build model and load checkpoint
    model = CoPhyloformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Support bare state_dict or wrapped checkpoint dict
    state_dict = checkpoint.get("model_state_dict", checkpoint) \
        if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=1)

    split_map = {
        "train": [("train", train_loader)],
        "val":   [("val",   val_loader)],
        "test":  [("test",  test_loader)],
        "all":   [("train", train_loader), ("val", val_loader), ("test", test_loader)],
    }
    splits_to_run = split_map[args.split]

    os.makedirs(args.save_dir, exist_ok=True)

    all_preds_list   = []
    all_targets_list = []

    for split_name, loader in splits_to_run:
        print(f"\n[INFO] Evaluating split: {split_name} ({len(loader.dataset)} samples)")
        preds_np, targets_np, metrics = evaluate_model(model, loader, device)

        print_metrics(metrics, split_name=split_name)

        # Save per-split artefacts if running multiple splits
        if len(splits_to_run) > 1:
            split_dir = os.path.join(args.save_dir, split_name)
        else:
            split_dir = args.save_dir

        save_metrics(metrics, split_dir)
        save_predictions(preds_np, targets_np, split_dir)

        if not args.no_plots:
            if MATPLOTLIB_AVAILABLE:
                plot_results(preds_np, targets_np, save_dir=split_dir)
            else:
                print("[WARNING] matplotlib not available — skipping plots.")

        all_preds_list.append(preds_np)
        all_targets_list.append(targets_np)

    # If 'all' splits, also save combined artefacts
    if args.split == "all" and len(splits_to_run) > 1:
        combined_preds   = np.concatenate(all_preds_list, axis=0)
        combined_targets = np.concatenate(all_targets_list, axis=0)
        combined_metrics = compute_metrics(combined_preds, combined_targets)
        combined_dir     = os.path.join(args.save_dir, "combined")
        print_metrics(combined_metrics, split_name="all splits combined")
        save_metrics(combined_metrics, combined_dir)
        save_predictions(combined_preds, combined_targets, combined_dir)
        if not args.no_plots and MATPLOTLIB_AVAILABLE:
            plot_results(combined_preds, combined_targets, save_dir=combined_dir)

    print(f"\n[INFO] All results saved under: {args.save_dir}\n")


if __name__ == "__main__":
    main()
