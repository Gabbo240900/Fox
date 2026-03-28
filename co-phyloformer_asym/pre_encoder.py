import torch, os
from data import CophylogenyDataset
from tqdm import tqdm

# src_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/Datasets'
# dst_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/test/'

src_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/Datasets/"
dst_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/asym_preencoded/"

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
    """Vectorised pairwise Jukes-Cantor distances. msa: [N, S] int tokens (PAD=22)."""
    valid = (msa != PAD_ID)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(0)         # [N, N, S]
    n_v = valid_pair.sum(dim=2).clamp(min=1).float()             # [N, N]
    mismatch = (msa.unsqueeze(1) != msa.unsqueeze(0)) & valid_pair
    p = mismatch.float().sum(dim=2) / n_v
    p = p.clamp(0.0, 0.74)
    dist = -0.75 * torch.log(1.0 - (4.0 / 3.0) * p)
    dist.fill_diagonal_(0.0)
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
