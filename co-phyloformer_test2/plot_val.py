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
        plt.savefig(f"{metric_name.lower()}_{name.replace('/', '_')}_combined.png")
        plt.show()

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
    plt.savefig(filename)
    plt.show()

def plot_labels_vs_predictions(train_labels, train_preds, val_labels, val_preds, event_name, filename):
    plt.figure(figsize=(8, 8))
    
    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)

    plt.scatter(train_labels, train_preds, alpha=0.6, label="Train", color="blue")
    plt.scatter(val_labels, val_preds, alpha=0.6, label="Validation", color="orange")

    # Handle empty train arrays
    arrays = []
    if train_labels.size > 0:
        arrays.append(train_labels)
    if train_preds.size > 0:
        arrays.append(train_preds)
    # Always include validation data
    arrays.append(val_labels)
    arrays.append(val_preds)

    # Compute global min/max safely
    global_min = min(arr.min() for arr in arrays)
    global_max = max(arr.max() for arr in arrays)

    plt.plot([global_min, global_max], [global_min, global_max], 'r--', label='Perfect Prediction')

    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title(f'Train vs Val — Labels vs Predictions for {event_name}')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()