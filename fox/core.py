"""Core prediction: a .tgl bundle -> event-frequency simplex."""

import warnings
from typing import Dict, List, Tuple

import torch

from .encoding import encode_msa, jukes_cantor_dist, EVENT_NAMES
from .io import read_tgl
from .model_loader import load_model

# Training host trees have 15-50 leaves (num_leaves in generate_data/generate_trees.py).
TRAIN_MAX_HOSTS = 50


def _looks_like_dna(msa: Dict[str, str]) -> bool:
    """True when nearly every non-gap letter is A, C, G, T/U or N."""
    letters = "".join(msa.values()).upper().replace("-", "").replace("?", "")
    if not letters:
        return False
    nuc = sum(letters.count(c) for c in "ACGTUN")
    return nuc / len(letters) > 0.9


def _check_input(host_msa: Dict[str, str], sym_msa: Dict[str, str]) -> None:
    for side, msa in (("host", host_msa), ("symbiont", sym_msa)):
        if _looks_like_dna(msa):
            warnings.warn(
                f"The {side} alignment looks like DNA. Fox reads amino acids only; "
                "translate it first with `python -m fox.translate`.",
                stacklevel=3,
            )
    if len(host_msa) > TRAIN_MAX_HOSTS:
        warnings.warn(
            f"{len(host_msa)} hosts: Fox was trained on at most {TRAIN_MAX_HOSTS} hosts, "
            "so this prediction is outside its training range.",
            stacklevel=3,
        )


def _infer(
    host_msa: Dict[str, str],
    sym_msa: Dict[str, str],
    mapping: List[Tuple[str, str]],
    model,
    device: str,
) -> Dict[str, float]:
    host_tok = encode_msa(host_msa)
    sym_tok = encode_msa(sym_msa)
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
) -> Dict[str, float]:
    """Predict relative cophylogenetic event frequencies from a .tgl file.

    A well-formatted .tgl bundles the host MSA, symbiont MSA, and the
    host<->symbiont leaf mapping, so this is the single-argument entrypoint.

    tgl_path  : path to the .tgl file.
    model     : preloaded Fox model (skip per-call load). Optional.
    ckpt_path, device : forwarded to load_model when model is None.
    drop_lost : simulated .tgl only; drop lost-gene leaves and the planted
                root row, as done for training (see fox.io.read_tgl).

    Returns {Speciation, HGT, Loss, Duplication} -> float, summing to 1.
    """
    d = read_tgl(tgl_path, drop_lost=drop_lost)
    _check_input(d["host_msa"], d["sym_msa"])
    if model is None:
        model = load_model(ckpt_path, device)
    return _infer(d["host_msa"], d["sym_msa"], d["mapping"], model, device)
