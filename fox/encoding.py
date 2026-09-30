"""Sequence encoding + Jukes-Cantor distances.

Shared by training/pre_encoder.py and inference so both encode identically.
"""

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_ID = 21
PAD_ID = 22
MAX_SEQ_LEN = 250
JC_MAX_P = 0.94  # protein JC saturates at p = 0.95

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]


def encode_sequence(sequence: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """One protein sequence -> [max_len] long tensor of AA indices (PAD-padded)."""
    encoded = [AA_TO_INDEX.get(aa, UNK_ID) for aa in sequence[:max_len]]
    encoded += [PAD_ID] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


def encode_msa(msa: dict) -> torch.Tensor:
    """Dict {name: sequence} -> [N, max_len] token tensor (row order = dict order)."""
    return torch.stack([encode_sequence(seq) for seq in msa.values()])


def jukes_cantor_dist(msa: torch.Tensor) -> torch.Tensor:
    """Vectorised pairwise Jukes-Cantor distances for proteins (20 states).

    msa: [N, S] int tokens (PAD=22). With p = fraction of mismatching sites,
    d = -(19/20) * ln(1 - (20/19) * p). p is capped just below the saturation
    point (19/20 = 0.95) so the log stays finite.
    """
    valid = (msa != PAD_ID)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(0)
    n_v = valid_pair.sum(dim=2).clamp(min=1).float()
    mismatch = (msa.unsqueeze(1) != msa.unsqueeze(0)) & valid_pair
    p = mismatch.float().sum(dim=2) / n_v
    p = p.clamp(0.0, JC_MAX_P)
    dist = -(19.0 / 20.0) * torch.log(1.0 - (20.0 / 19.0) * p)
    dist.fill_diagonal_(0.0)
    return dist
