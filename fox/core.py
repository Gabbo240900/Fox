"""Core prediction: a .tgl bundle -> event-frequency simplex."""

import warnings
from typing import Dict, List, Optional, Tuple

import torch

from .encoding import encode_msa, jukes_cantor_dist, EVENT_NAMES, MAX_SEQ_LEN
from .io import read_tgl
from .model_loader import load_model

# Training host trees have 15-50 leaves (num_leaves in generate_data/generate_trees.py).
TRAIN_MAX_HOSTS = 50
# Training symbiont trees are capped at 142 leaves (PARA_MAX_LEAVES in training/train.py).
TRAIN_MAX_SYMBIONTS = 142
# Training alignments have exactly MAX_SEQ_LEN columns.
TRAIN_MAX_COLUMNS = MAX_SEQ_LEN


def _looks_like_dna(msa: Dict[str, str]) -> bool:
    """True when nearly every non-gap letter is A, C, G, T/U or N."""
    letters = "".join(msa.values()).upper().replace("-", "").replace("?", "")
    if not letters:
        return False
    nuc = sum(letters.count(c) for c in "ACGTUN")
    return nuc / len(letters) > 0.9


def _check_input(
    host_msa: Dict[str, str],
    sym_msa: Dict[str, str],
    max_columns: Optional[int] = None,
) -> None:
    for side, msa in (("host", host_msa), ("symbiont", sym_msa)):
        if _looks_like_dna(msa):
            warnings.warn(
                f"The {side} alignment looks like DNA. Fox reads amino acids only; "
                "translate it first with `python -m fox.translate`.",
                stacklevel=3,
            )
        n_cols = max(len(seq) for seq in msa.values())
        used = n_cols if max_columns is None else min(n_cols, max_columns)
        if used < n_cols:
            warnings.warn(
                f"The {side} alignment has {n_cols} columns; reading only the first "
                f"{used} (--max-columns / max_columns=).",
                stacklevel=3,
            )
        if used > TRAIN_MAX_COLUMNS:
            warnings.warn(
                f"Reading {used} columns of the {side} alignment: Fox was trained on "
                f"{TRAIN_MAX_COLUMNS} columns, so this prediction is outside its training range. "
                "Time and memory grow with the number of columns; --max-columns caps them.",
                stacklevel=3,
            )
    if len(host_msa) > TRAIN_MAX_HOSTS:
        warnings.warn(
            f"{len(host_msa)} hosts: Fox was trained on at most {TRAIN_MAX_HOSTS} hosts, "
            "so this prediction is outside its training range.",
            stacklevel=3,
        )
    if len(sym_msa) > TRAIN_MAX_SYMBIONTS:
        warnings.warn(
            f"{len(sym_msa)} symbionts: Fox was trained on at most {TRAIN_MAX_SYMBIONTS} symbionts, "
            "so this prediction is outside its training range.",
            stacklevel=3,
        )


def _cap(msa: Dict[str, str], max_columns: Optional[int]) -> Optional[int]:
    """max_columns, but never more than the longest sequence of this alignment."""
    if max_columns is None:
        return None
    return min(max_columns, max(len(seq) for seq in msa.values()))


def _infer(
    host_msa: Dict[str, str],
    sym_msa: Dict[str, str],
    mapping: List[Tuple[str, str]],
    model,
    device: str,
    max_columns: Optional[int] = None,
) -> Dict[str, float]:
    # Each side stops at its own length: a max_columns above it would only add
    # PAD columns, which shift the prediction slightly when there are thousands.
    host_tok = encode_msa(host_msa, _cap(host_msa, max_columns))
    sym_tok = encode_msa(sym_msa, _cap(sym_msa, max_columns))
    host_dist = jukes_cantor_dist(host_tok)
    sym_dist = jukes_cantor_dist(sym_tok)

    h_idx = {n: i for i, n in enumerate(host_msa)}
    s_idx = {n: i for i, n in enumerate(sym_msa)}
    mapped = [(h_idx[h], s_idx[s]) for h, s in mapping if h in h_idx and s in s_idx]
    if not mapped:
        raise ValueError(
            "No mapping pair matched the alignment leaf names. "
            "Check the DISTRIBUTION block against the ALIGNMENT leaf ids."
        )

    with torch.no_grad():
        out = model(
            host_tok.unsqueeze(0).to(device),
            sym_tok.unsqueeze(0).to(device),
            [mapped],
            host_dist=host_dist.unsqueeze(0).to(device),
            para_dist=sym_dist.unsqueeze(0).to(device),
        )  # [1, 4], already softmaxed

    return dict(zip(EVENT_NAMES, out.squeeze(0).cpu().tolist()))


def predict_tgl(
    tgl_path: str,
    model=None,
    ckpt_path: str = None,
    device: str = "cpu",
    drop_lost: bool = False,
    max_columns: Optional[int] = None,
) -> Dict[str, float]:
    """Predict relative cophylogenetic event frequencies from a .tgl file.

    A well-formatted .tgl bundles the host MSA, symbiont MSA, and the
    host<->symbiont leaf mapping, so this is the single-argument entrypoint.

    tgl_path  : path to the .tgl file.
    model     : preloaded Fox model (skip per-call load). Optional.
    ckpt_path, device : forwarded to load_model when model is None.
    drop_lost : simulated .tgl only; drop lost-gene leaves and the planted
                root row, as done for training (see fox.io.read_tgl).
    max_columns : read only the first max_columns alignment columns
                  (default None: the whole alignment). More than 250
                  columns is outside the training range and triggers a
                  warning; time and memory grow with the columns read.

    Returns {Speciation, HGT, Loss, Duplication} -> float, summing to 1.
    """
    d = read_tgl(tgl_path, drop_lost=drop_lost)
    _check_input(d["host_msa"], d["sym_msa"], max_columns)
    if model is None:
        model = load_model(ckpt_path, device)
    return _infer(d["host_msa"], d["sym_msa"], d["mapping"], model, device, max_columns)
