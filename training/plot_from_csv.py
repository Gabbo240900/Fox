import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]
EVENT_LABELS = {"Speciation": "Cospeciation", "HGT": "Host switch", "Loss": "Loss", "Duplication": "Duplication"}
def disp(e): return EVENT_LABELS.get(e, e)  # display label only; data keys/filenames stay "Speciation" etc


def load_csv(path):
    """Load a prediction CSV written by write_prediction_csv.

    Returns (preds, labels) as float32 numpy arrays of shape (N, 4).
    Expects columns: true_Speciation, ..., pred_Speciation, ...
    """
    path = Path(path)
    preds, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append([float(row[f"true_{n}"]) for n in EVENT_NAMES])
            preds.append([float(row[f"pred_{n}"]) for n in EVENT_NAMES])
    return np.array(preds, dtype=np.float32), np.array(labels, dtype=np.float32)


def plot_scatter(train_preds, train_labels, val_preds, val_labels, output_dir):
    """One scatter plot per event saved under output_dir/scatter_plots/."""
    out = Path(output_dir) / "scatter_plots"
    out.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(EVENT_NAMES):
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(train_labels[:, i], train_preds[:, i],
                   s=12, alpha=0.4, color="#4C72B0", label="Train")
        ax.scatter(val_labels[:, i], val_preds[:, i],
                   s=12, alpha=0.6, color="#E24A33", label="Validation")

        all_vals = np.concatenate([train_labels[:, i], train_preds[:, i],
                                    val_labels[:, i],   val_preds[:, i]])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.8, label="Perfect prediction")

        ax.set_xlabel("True", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{disp(name)} — Scatter (Train vs Val)", fontsize=13)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(out / f"scatter_{name}.png", dpi=300)
        plt.close(fig)


def plot_density(train_preds, train_labels, val_preds, val_labels, output_dir, bins=40):
    """2D density (hexbin) plot per event saved under output_dir/density_plots/."""
    out = Path(output_dir) / "density_plots"
    out.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(EVENT_NAMES):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, preds, labels, title, cmap in [
            (axes[0], train_preds, train_labels, "Train",      "Blues"),
            (axes[1], val_preds,   val_labels,   "Validation", "Reds"),
        ]:
            hb = ax.hexbin(labels[:, i], preds[:, i],
                           gridsize=bins, cmap=cmap, mincnt=1, linewidths=0.2)
            fig.colorbar(hb, ax=ax, label="Count")

            all_vals = np.concatenate([labels[:, i], preds[:, i]])
            lo, hi = float(all_vals.min()), float(all_vals.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.8, label="Perfect prediction")

            ax.set_xlabel("True", fontsize=11)
            ax.set_ylabel("Predicted", fontsize=11)
            ax.set_title(f"{disp(name)} — {title}", fontsize=12)
            ax.legend(frameon=True, fontsize=9)
            ax.grid(True, alpha=0.2, linestyle="--")

        fig.suptitle(f"{disp(name)} — Prediction density", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"density_{name}.png", dpi=300)
        plt.close(fig)
