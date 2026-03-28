import torch, os
from data import CophylogenyDataset
from tqdm import tqdm

src_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/Datasets'
dst_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/test/'

# src_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/Datasets/"
# dst_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/asym_preencoded/"

os.makedirs(dst_dir, exist_ok=True)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID = 21
PAD_ID = 22


def encode_sequence(sequence, max_len=200):
    encoded = [AA_TO_INDEX.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


def jukes_cantor_dist(msa: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Jukes-Cantor distances from a padded MSA token tensor.

    msa: [N, S]  int token ids  (PAD = 22)
    Returns: [N, N] float distance matrix (symmetric, zero diagonal).

    Runs on CPU inside collate workers — the loop over pairs is fast for
    typical N ≤ 50 leaves.  The distance is clamped before log to avoid NaN.
    """
    N, S = msa.shape
    dist = torch.zeros(N, N, dtype=torch.float32)
    valid = (msa != 22)                  # [N, S]  True = real residue
    for i in range(N):
        for j in range(i + 1, N):
            v = valid[i] & valid[j]      # [S]
            n_v = v.sum().clamp(min=1).float()
            p   = ((msa[i] != msa[j]) & v).float().sum() / n_v
            p   = p.clamp(0.0, 0.74)    # keep argument of log > 0
            d   = -0.75 * torch.log(1.0 - (4.0 / 3.0) * p)
            dist[i, j] = dist[j, i] = d
    return dist


dataset = CophylogenyDataset(src_dir).get_data()

for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
    if len(sample["host_msas"]) == 0 or len(sample["parasite_msas"]) == 0:
        continue

    host_tokens = torch.stack([encode_sequence(seq) for seq in sample["host_msas"].values()])
    para_tokens = torch.stack([encode_sequence(seq) for seq in sample["parasite_msas"].values()])

    sample["host_dist"] = jukes_cantor_dist(host_tokens)   # [N_h, N_h]
    sample["para_dist"] = jukes_cantor_dist(para_tokens)   # [N_p, N_p]

    out_path = os.path.join(dst_dir, f"{i:07d}.pt")
    torch.save(sample, out_path)
