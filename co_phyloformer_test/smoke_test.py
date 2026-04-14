"""
Minimal smoke test: data loading → model forward → loss, no Fabric/distributed.
Run from co_phyloformer_test/: python smoke_test.py
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import Cophyloformer

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "generate_treeducken", "generated_trees", "test")
BATCH_SIZE  = 4
N_BATCHES   = 3
EVENT_NAMES = ["Cospeciations", "Host_spread/Switches"]
HOST_MAX    = 50
PARA_MAX    = 64

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID, PAD_ID = 21, 22


def encode_sequence(seq, max_len=128):
    enc = [AA_TO_INDEX.get(aa, UNK_ID) for aa in seq[:max_len]]
    enc += [PAD_ID] * (max_len - len(enc))
    return torch.tensor(enc, dtype=torch.long)


class Dataset_(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

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


# ── dataset ───────────────────────────────────────────────────────────────────
print(f"Data dir: {os.path.abspath(DATA_DIR)}")
ds = Dataset_(DATA_DIR)
print(f"  {len(ds)} samples")

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ── model (small dims for fast CPU run) ───────────────────────────────────────
model = Cophyloformer(
    hidden_dim=64, pair_dim=16, cls_dim=128, axial_layers=1,
    num_cross_layers=1, num_events=len(EVENT_NAMES),
)
print(f"  model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── forward + backward ────────────────────────────────────────────────────────
model.train()
for i, batch in enumerate(loader):
    if i >= N_BATCHES:
        break
    if batch is None:
        print(f"  batch {i}: empty, skipped"); continue

    out = model(batch["host_msa"], batch["parasite_msa"], batch["mappings"], batch["sim_time"])
    loss = F.mse_loss(out, batch["labels"])
    loss.backward()

    print(f"  batch {i} | host {tuple(batch['host_msa'].shape)} | para {tuple(batch['parasite_msa'].shape)} "
          f"| labels {batch['labels'].tolist()} | out {out.detach().tolist()} | loss {loss.item():.6f}")

print("\nSmoke test passed.")
