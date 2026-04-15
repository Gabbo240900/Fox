"""
End-to-end smoke test: data loading → forward → loss → backward → CSV + scatter plots.
Run from co_phyloformer_test/: python smoke_test.py
"""
import sys, os, glob, csv, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Cophyloformer

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generate_treeducken", "generated_trees", "test")
EVENT_NAMES = ["Cospeciations", "Host_spread/Switches"]
HOST_MAX    = 50
PARA_MAX    = 64
BATCH_SIZE  = 4
N_TRAIN_BATCHES = 3

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID, PAD_ID = 21, 22


def encode_sequence(seq, max_len=128):
    enc = [AA_TO_INDEX.get(aa, UNK_ID) for aa in seq[:max_len]]
    enc += [PAD_ID] * (max_len - len(enc))
    return torch.tensor(enc, dtype=torch.long)


class TDDataset(Dataset):
    def __init__(self, data_dir, pt_files=None):
        self.files = pt_files if pt_files is not None else sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        if not s.get("host_msas") or not s.get("parasite_msas"):
            return None
        h_idx = {n: i for i, n in enumerate(s["host_msas"])}
        p_idx = {n: i for i, n in enumerate(s["parasite_msas"])}
        return {
            "host_msa":     torch.stack([encode_sequence(v) for v in s["host_msas"].values()]),
            "parasite_msa": torch.stack([encode_sequence(v) for v in s["parasite_msas"].values()]),
            "mappings":     [(h_idx[h], p_idx[p]) for p, h in s["mappings"] if p in p_idx and h in h_idx],
            "labels":       torch.tensor([s["event_frequencies"].get(e, 0.0) for e in EVENT_NAMES], dtype=torch.float32),
            "sim_time":     torch.tensor([s["event_frequencies"].get("Sim_time", 1.0)], dtype=torch.float32),
        }


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    def pad(msas, cap):
        n = min(max(m.shape[0] for m in msas), cap)
        return torch.stack([F.pad(m[:n], (0, 0, 0, max(0, n - m.shape[0])), value=PAD_ID) for m in msas])
    return {
        "host_msa":     pad([s["host_msa"]     for s in batch], HOST_MAX),
        "parasite_msa": pad([s["parasite_msa"] for s in batch], PARA_MAX),
        "labels":       torch.stack([s["labels"]   for s in batch]),
        "mappings":     [s["mappings"]             for s in batch],
        "sim_time":     torch.stack([s["sim_time"] for s in batch]),
    }


def compute_predictions(model, loader, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            out = model(
                batch["host_msa"].to(device),
                batch["parasite_msa"].to(device),
                batch["mappings"],
                batch["sim_time"].to(device),
            )
            preds.append(out.cpu())
            labels.append(batch["labels"])
    if not preds:
        return None, None
    return torch.cat(preds), torch.cat(labels)


# ── dataset / train-val split ─────────────────────────────────────────────────
print(f"Data dir: {os.path.abspath(DATA_DIR)}")
dataset = TDDataset(DATA_DIR)
print(f"  {len(dataset)} samples")

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_sub = Subset(dataset, train_idx)
val_sub   = Subset(dataset, val_idx)
print(f"  train={len(train_sub)}  val={len(val_sub)}")

train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ── model ─────────────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = Cophyloformer(
    hidden_dim=64, pair_dim=16, cls_dim=128, axial_layers=1,
    num_cross_layers=1, num_events=len(EVENT_NAMES),
).to(device)
print(f"  model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ── training pass ─────────────────────────────────────────────────────────────
print("\n── Training pass ──")
model.train()
for i, batch in enumerate(train_loader):
    if i >= N_TRAIN_BATCHES:
        break
    if batch is None:
        continue
    optimizer.zero_grad()
    out  = model(batch["host_msa"], batch["parasite_msa"], batch["mappings"], batch["sim_time"])
    loss = F.mse_loss(out, batch["labels"])
    loss.backward()
    optimizer.step()
    print(f"  batch {i} | loss {loss.item():.6f}")

# ── val predictions → CSV ─────────────────────────────────────────────────────
print("\n── Val predictions + CSV ──")
val_preds, val_labels = compute_predictions(model, val_loader, device)
assert val_preds is not None, "compute_predictions returned None — check val loader"
print(f"  val_preds shape:  {tuple(val_preds.shape)}")
print(f"  val_labels shape: {tuple(val_labels.shape)}")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_predictions.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"true_{n}" for n in EVENT_NAMES] + [f"pred_{n}" for n in EVENT_NAMES])
    for true_row, pred_row in zip(val_labels.tolist(), val_preds.tolist()):
        writer.writerow(true_row + pred_row)
print(f"  saved {csv_path}  ({len(val_labels)} rows)")

# ── scatter plots ─────────────────────────────────────────────────────────────
print("\n── Scatter plots ──")
scatter_n   = min(50, len(train_idx))
scatter_idx = random.sample(train_idx, scatter_n)
train_scatter_ds = TDDataset(DATA_DIR, pt_files=[dataset.files[i] for i in scatter_idx])
train_loader_sc  = DataLoader(train_scatter_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
train_preds, train_labels = compute_predictions(model, train_loader_sc, device)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scatter_plots")
os.makedirs(save_dir, exist_ok=True)
for i, name in enumerate(EVENT_NAMES):
    safe_name = name.replace("/", "_")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(train_labels[:, i].numpy(), train_preds[:, i].numpy(), alpha=0.4, s=10, color="blue",   label="Train")
    ax.scatter(val_labels[:, i].numpy(),   val_preds[:, i].numpy(),   alpha=0.4, s=10, color="orange", label="Validation")
    lo = min(train_labels[:, i].min(), val_labels[:, i].min())
    hi = max(train_labels[:, i].max(), val_labels[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect prediction")
    ax.set_xlabel("True Labels"); ax.set_ylabel("Predictions")
    ax.set_title(f"Train vs Val — {name}"); ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    png_path = os.path.join(save_dir, f"{safe_name}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  saved {png_path}")

print("\nAll checks passed ✓")
