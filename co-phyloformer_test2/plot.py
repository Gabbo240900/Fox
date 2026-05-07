import matplotlib.pyplot as plt
import numpy as np

def plot_event_metric_over_epochs(train_metrics, val_metrics, event_names, metric_name):
    train_np = np.array(train_metrics).T
    val_np = np.array(val_metrics).T
    for i, name in enumerate(event_names):
        plt.figure(figsize=(20, 5))
        plt.plot(range(1, len(train_np[i]) + 1), train_np[i], marker='o', linestyle='-', label=f'{name} Train {metric_name}')
        plt.plot(range(1, len(val_np[i]) + 1), val_np[i], marker='o', linestyle='-', label=f'{name} Val {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{name} - Train vs Val {metric_name}')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(range(0, len(train_np[i]) + 1, 10))
        plt.tight_layout()
        plt.savefig(f"{metric_name.lower()}_{name.replace('/', '_')}_combined.png", dpi=300)
        plt.close()

def plot_epoch_loss_curve(train_losses, val_losses, filename, use_log_scale=True):
    plt.figure(figsize=(12, 6))
    offset = 1e-8

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    if use_log_scale:
        train_losses = np.log10(train_losses + offset)
        val_losses = np.log10(val_losses + offset)

    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', label="Validation Loss")

    plt.xlabel('Epoch')
    plt.ylabel('log10(Loss)' if use_log_scale else 'Loss')
    plt.title('Training vs Validation Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_labels_vs_predictions(train_labels, train_preds, val_labels, val_preds, event_name, filename):
    plt.figure(figsize=(8, 8))
    
    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)

    plt.scatter(train_labels, train_preds, alpha=0.6, label="Train", color="blue")
    plt.scatter(val_labels, val_preds, alpha=0.6, label="Validation", color="orange")

    min_val = min(train_labels.min(), val_labels.min(), train_preds.min(), val_preds.min())
    max_val = max(train_labels.max(), val_labels.max(), train_preds.max(), val_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title(f'Train vs Val — Labels vs Predictions for {event_name}')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_interval_q50_q90(
    labels,
    q50,
    q90,
    event_name,
    filename="interval_q50_q90.png",
    max_points=8000,
):
    labels = np.asarray(labels).astype(float)
    q50 = np.asarray(q50).astype(float)
    q90 = np.asarray(q90).astype(float)

    # subsample for readability
    n = len(labels)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        labels = labels[idx]
        q50 = q50[idx]
        q90 = q90[idx]

    # sort by GT for cleaner structure
    order = np.argsort(labels)
    labels = labels[order]
    q50 = q50[order]
    q90 = q90[order]

    plt.figure(figsize=(8, 8))

    # Brighter interval color
    interval_color = "#4C72B0"  # clean blue
    median_color = "#1f77b4"    # strong blue
    gt_color = "#E24A33"        # bright orange-red

    # interval bars (thicker + clearer)
    plt.vlines(
        labels,
        q50,
        q90,
        color=interval_color,
        alpha=0.35,
        linewidth=1.2,
        label="Pred interval [q50, q90]"
    )

    # median predictions
    plt.scatter(
        labels,
        q50,
        s=18,
        alpha=0.7,
        color=median_color,
        edgecolor="white",
        linewidth=0.3,
        label="Pred q50"
    )

    # GT diagonal points
    plt.scatter(
        labels,
        labels,
        s=18,
        alpha=0.8,
        color=gt_color,
        edgecolor="white",
        linewidth=0.3,
        label="GT (y=x)"
    )

    # perfect prediction reference line
    mx = max(1e-6, float(np.max(labels)))
    plt.plot(
        [0, mx],
        [0, mx],
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.8,
        label="Perfect prediction"
    )

    plt.xlabel("True Labels", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(f"{event_name} — Predicted interval [q50, q90] vs GT", fontsize=13)

    plt.legend(frameon=True)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()