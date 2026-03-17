"""
dataset.py — Data loading and encoding for Co-Phyloformer.

Parses .tgl (NEXUS-like) files produced by asymmetree, encodes amino-acid
sequences as integer tensors, and wraps everything in a PyTorch Dataset /
DataLoader pipeline.
"""

import os
import re
import glob
import random
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

AA_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]  # 22 tokens
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}

# ---------------------------------------------------------------------------
# Sequence encoder
# ---------------------------------------------------------------------------

def encode_sequence(seq: str, max_seq_len: Optional[int] = None) -> LongTensor:
    """Encode an amino-acid string into an integer LongTensor.

    Characters not present in AA_VOCAB are mapped to the index of 'X'.
    Optionally truncates or pads (with 'X' index) to *max_seq_len*.

    Args:
        seq: Amino-acid string (gaps '-' are kept).
        max_seq_len: If given, truncate to this length or right-pad with the
                     'X' index so the output tensor always has this length.

    Returns:
        LongTensor of shape (len,).
    """
    seq = seq.upper()
    x_idx = AA_TO_IDX["X"]
    indices = [AA_TO_IDX.get(ch, x_idx) for ch in seq]

    if max_seq_len is not None:
        if len(indices) >= max_seq_len:
            indices = indices[:max_seq_len]
        else:
            indices += [x_idx] * (max_seq_len - len(indices))

    return torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------------
# .tgl parser
# ---------------------------------------------------------------------------

def _parse_alignment_block(block_text: str) -> Dict[str, str]:
    """Parse the content between the opening quote and closing quote of an
    ALIGNMENT block.

    Each non-empty line inside the quotes has the form:
        [optional whitespace] <LEAF_ID>  <SEQUENCE>
    where LEAF_ID and SEQUENCE are separated by two or more spaces.
    """
    seqs: Dict[str, str] = {}
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Split on 2+ whitespace characters separating ID from sequence
        parts = re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 2:
            leaf_id, sequence = parts
            seqs[leaf_id.strip()] = sequence.strip()
    return seqs


def _parse_range_block(range_text: str) -> List[Tuple[str, str]]:
    """Parse RANGE lines of the form ``P_id: H_id`` into a list of tuples."""
    mappings: List[Tuple[str, str]] = []
    for raw_line in range_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s*:\s*(\S+)$", line)
        if m:
            mappings.append((m.group(1), m.group(2)))
    return mappings


def parse_tgl(filepath: str) -> dict:
    """Parse a .tgl file and return a structured dictionary.

    Args:
        filepath: Absolute path to the .tgl file.

    Returns:
        Dictionary with keys:
            host_seqs      : {leaf_id: sequence_str}
            parasite_seqs  : {leaf_id: sequence_str}
            mappings       : [(parasite_id, host_id), ...]
            labels         : {speciation_freq, hgt_freq, loss_freq, duplication_freq}
            metadata       : {host_num_leaves, symbiont_num_leaves, sim_time}

    Raises:
        ValueError: If a required section is missing or unparseable.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        text = fh.read()

    # ---- Host alignment ------------------------------------------------
    host_align_match = re.search(
        r"BEGIN HOST;.*?ALIGNMENT\s+\*\s+\w+\s*=\s*'(.*?)'",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not host_align_match:
        raise ValueError(f"Cannot find HOST ALIGNMENT block in {filepath}")
    host_seqs = _parse_alignment_block(host_align_match.group(1))

    # ---- Parasite alignment --------------------------------------------
    para_align_match = re.search(
        r"BEGIN PARASITE;.*?ALIGNMENT\s+\*\s+\w+\s*=\s*'(.*?)'",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not para_align_match:
        raise ValueError(f"Cannot find PARASITE ALIGNMENT block in {filepath}")
    parasite_seqs = _parse_alignment_block(para_align_match.group(1))

    # ---- Distribution / Range ------------------------------------------
    range_match = re.search(
        r"BEGIN DISTRIBUTION;\s*RANGE(.*?)END\s*;",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not range_match:
        raise ValueError(f"Cannot find DISTRIBUTION RANGE block in {filepath}")
    mappings = _parse_range_block(range_match.group(1))

    # ---- Scalar statistics at the bottom of the file -------------------
    def _float(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def _int(pattern: str) -> Optional[int]:
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    speciation_freq = _float(r"Speciation_freq\s*:\s*([0-9.eE+\-]+)")
    hgt_freq = _float(r"HGT_freq\s*:\s*([0-9.eE+\-]+)")
    loss_freq = _float(r"Loss_freq\s*:\s*([0-9.eE+\-]+)")
    dup_freq = _float(r"Duplication_freq\s*:\s*([0-9.eE+\-]+)")

    host_leaves = _int(r"Host_num_leaves\s*:\s*(\d+)")
    symbiont_leaves = _int(r"Symbiont_num_leaves\s*:\s*(\d+)")
    sim_time = _float(r"Sim_time\s*:\s*([0-9.eE+\-]+)")

    labels = {
        "speciation_freq": speciation_freq,
        "hgt_freq": hgt_freq,
        "loss_freq": loss_freq,
        "duplication_freq": dup_freq,
    }
    metadata = {
        "host_num_leaves": host_leaves,
        "symbiont_num_leaves": symbiont_leaves,
        "sim_time": sim_time,
    }

    return {
        "host_seqs": host_seqs,
        "parasite_seqs": parasite_seqs,
        "mappings": mappings,
        "labels": labels,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CoPhyloDataset(Dataset):
    """PyTorch Dataset over a collection of .tgl files.

    Each call to ``__getitem__`` returns a single parsed sample with sequences
    encoded as integer tensors.

    Args:
        data_dir:    Path to the Datasets/ folder containing Dataset*.tgl files.
        indices:     1-based list of dataset indices to load.  If *None*, all
                     Dataset*.tgl files found in *data_dir* are used.
        max_seq_len: Truncate / pad every sequence to exactly this many
                     positions.  Use *None* to keep variable lengths.
    """

    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        max_seq_len: int = 512,
    ) -> None:
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        if indices is not None:
            candidates = [
                os.path.join(data_dir, f"Dataset{i}.tgl") for i in indices
            ]
        else:
            pattern = os.path.join(data_dir, "Dataset*.tgl")
            candidates = sorted(
                glob.glob(pattern),
                key=lambda p: int(re.search(r"Dataset(\d+)\.tgl", p).group(1)),
            )

        # Validate files exist; warn and skip missing/malformed ones
        self.file_paths: List[str] = []
        for fp in candidates:
            if not os.path.isfile(fp):
                warnings.warn(f"File not found, skipping: {fp}")
            else:
                self.file_paths.append(fp)

        if len(self.file_paths) == 0:
            warnings.warn(f"No valid .tgl files found in {data_dir!r}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load and encode one dataset sample.

        Returns:
            dict with keys:
                host_seqs     : {leaf_id: LongTensor[seq_len]}
                parasite_seqs : {leaf_id: LongTensor[seq_len]}
                mappings      : list of (parasite_id, host_id) tuples
                labels        : FloatTensor[4]  — [spec_freq, hgt_freq,
                                                    loss_freq, dup_freq]
                metadata      : dict of scalar values
                filepath      : source file path (str)
        """
        fp = self.file_paths[idx]
        try:
            parsed = parse_tgl(fp)
        except Exception as exc:
            warnings.warn(f"Failed to parse {fp}: {exc}")
            # Return an empty sentinel so the caller can detect bad samples
            return {
                "host_seqs": {},
                "parasite_seqs": {},
                "mappings": [],
                "labels": torch.zeros(4, dtype=torch.float32),
                "metadata": {},
                "filepath": fp,
            }

        host_seqs_encoded = {
            leaf: encode_sequence(seq, self.max_seq_len)
            for leaf, seq in parsed["host_seqs"].items()
        }
        parasite_seqs_encoded = {
            leaf: encode_sequence(seq, self.max_seq_len)
            for leaf, seq in parsed["parasite_seqs"].items()
        }

        lbl = parsed["labels"]
        labels_tensor = torch.tensor(
            [
                lbl["speciation_freq"] if lbl["speciation_freq"] is not None else 0.0,
                lbl["hgt_freq"] if lbl["hgt_freq"] is not None else 0.0,
                lbl["loss_freq"] if lbl["loss_freq"] is not None else 0.0,
                lbl["duplication_freq"] if lbl["duplication_freq"] is not None else 0.0,
            ],
            dtype=torch.float32,
        )

        return {
            "host_seqs": host_seqs_encoded,
            "parasite_seqs": parasite_seqs_encoded,
            "mappings": parsed["mappings"],
            "labels": labels_tensor,
            "metadata": parsed["metadata"],
            "filepath": fp,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> List[dict]:
    """Collate a list of samples into a batch.

    Because each sample contains trees of varying size (different number of
    leaves) and potentially varying sequence length, we *do not* stack tensors
    across samples.  Instead we:

    1. Determine the maximum sequence length across all sequences in the batch.
    2. Pad every individual sequence tensor to that common length (using the
       'X' token index).
    3. Return the batch as a plain Python list of dicts (one entry per sample),
       with padded tensors.

    This lets a DataLoader with batch_size > 1 group samples together while
    downstream code can still iterate over the list and handle variable tree
    topology.

    Args:
        batch: List of dicts as returned by ``CoPhyloDataset.__getitem__``.

    Returns:
        List of dicts with sequences padded to a common length within the batch.
    """
    if not batch:
        return []

    x_idx = AA_TO_IDX["X"]

    # Find the maximum sequence length present in this batch
    max_len = 0
    for sample in batch:
        for t in sample["host_seqs"].values():
            max_len = max(max_len, t.shape[0])
        for t in sample["parasite_seqs"].values():
            max_len = max(max_len, t.shape[0])

    if max_len == 0:
        return list(batch)

    def _pad(tensor: LongTensor, length: int) -> LongTensor:
        cur = tensor.shape[0]
        if cur >= length:
            return tensor[:length]
        pad = torch.full((length - cur,), x_idx, dtype=torch.long)
        return torch.cat([tensor, pad], dim=0)

    processed = []
    for sample in batch:
        new_host = {k: _pad(v, max_len) for k, v in sample["host_seqs"].items()}
        new_para = {k: _pad(v, max_len) for k, v in sample["parasite_seqs"].items()}
        processed.append(
            {
                "host_seqs": new_host,
                "parasite_seqs": new_para,
                "mappings": sample["mappings"],
                "labels": sample["labels"],
                "metadata": sample["metadata"],
                "filepath": sample["filepath"],
            }
        )
    return processed


# ---------------------------------------------------------------------------
# Train / Val / Test split utility
# ---------------------------------------------------------------------------

def _discover_indices(data_dir: str, manifest_file: Optional[str] = None) -> List[int]:
    """Return sorted list of dataset indices found in data_dir.

    For 1M+ datasets, passing a pre-built ``manifest_file`` (one absolute path
    per line) is much faster than globbing the directory.  Build it once with:

        ls /path/to/Datasets/Dataset*.tgl | sort -V > manifest.txt

    If ``manifest_file`` is None the function falls back to glob, which is fine
    for up to ~100 k files but can be slow beyond that.
    """
    if manifest_file is not None and os.path.isfile(manifest_file):
        indices = []
        with open(manifest_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                m = re.search(r"Dataset(\d+)\.tgl", line)
                if m:
                    indices.append(int(m.group(1)))
        return sorted(indices)

    pattern = os.path.join(data_dir, "Dataset*.tgl")
    found_files = glob.glob(pattern)
    if not found_files:
        raise ValueError(f"No Dataset*.tgl files found in {data_dir!r}")
    return sorted(
        int(re.search(r"Dataset(\d+)\.tgl", os.path.basename(fp)).group(1))
        for fp in found_files
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    max_seq_len: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    rank: int = 0,
    world_size: int = 1,
    manifest_file: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders.

    DDP-aware: when ``world_size > 1`` a ``DistributedSampler`` is used for the
    training set so that each GPU sees a non-overlapping subset of the data.
    Validation and test sets run on **all** ranks (with their own samplers) so
    that per-rank metrics can be all_reduced in the training script.

    For 1M+ datasets pass ``manifest_file`` pointing to a pre-built index file
    to avoid slow directory globbing.

    Args:
        data_dir:           Path to the Datasets/ folder.
        batch_size:         Samples per batch *per GPU*.
        train_frac:         Fraction of data used for training.
        val_frac:           Fraction of data used for validation.
        seed:               Random seed for reproducible splits.
        max_seq_len:        Sequence truncation / padding length.
        num_workers:        DataLoader worker processes per GPU.
        pin_memory:         Pin host memory for faster GPU transfers (requires CUDA).
        persistent_workers: Keep worker processes alive between epochs.
        prefetch_factor:    Batches to prefetch per worker.
        rank:               This process's global rank (0 for single-GPU).
        world_size:         Total number of processes (1 for single-GPU).
        manifest_file:      Path to pre-built file list (one path per line).

    Returns:
        (train_loader, val_loader, test_loader) tuple.
    """
    if train_frac + val_frac > 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0"
        )

    all_indices = _discover_indices(data_dir, manifest_file)
    n = len(all_indices)

    rng = random.Random(seed)
    shuffled = all_indices[:]
    rng.shuffle(shuffled)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_indices = shuffled[:n_train]
    val_indices   = shuffled[n_train: n_train + n_val]
    test_indices  = shuffled[n_train + n_val:]

    train_ds = CoPhyloDataset(data_dir, indices=train_indices, max_seq_len=max_seq_len)
    val_ds   = CoPhyloDataset(data_dir, indices=val_indices,   max_seq_len=max_seq_len)
    test_ds  = CoPhyloDataset(data_dir, indices=test_indices,  max_seq_len=max_seq_len)

    # Workers / memory settings — disable persistent_workers when num_workers=0
    _persistent = persistent_workers and num_workers > 0
    _prefetch   = prefetch_factor if num_workers > 0 else None

    # Worker seed initialisation for reproducibility across ranks
    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + rank * 1000 + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    common_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        worker_init_fn=_worker_init,
    )

    if world_size > 1:
        # DistributedSampler handles data sharding across GPUs.
        # drop_last=True on training avoids uneven last batches across ranks.
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        train_loader = DataLoader(train_ds, sampler=train_sampler, **common_kwargs)
        val_loader   = DataLoader(val_ds,   sampler=val_sampler,   **common_kwargs)
        test_loader  = DataLoader(test_ds,  sampler=test_sampler,  **common_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True,  **common_kwargs)
        val_loader   = DataLoader(val_ds,   shuffle=False, **common_kwargs)
        test_loader  = DataLoader(test_ds,  shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
