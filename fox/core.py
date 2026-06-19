"""Core prediction: a .tgl bundle -> event-frequency simplex."""

from typing import Dict, List, Optional, Tuple

import torch

from .encoding import encode_msa, jukes_cantor_dist, EVENT_NAMES
from .io import read_tgl
from .model_loader import load_model


def _infer(
    host_msa: Dict[str, str],
    sym_msa: Dict[str, str],
    mapping: List[Tuple[str, str]],
    sim_time: Optional[float],
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

    t = None
    if sim_time is not None:
        t = torch.tensor([[float(sim_time)]], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(
            host_tok.unsqueeze(0).to(device),
            sym_tok.unsqueeze(0).to(device),
            [mapped],
            t,
            host_dist=host_dist.unsqueeze(0).to(device),
            para_dist=sym_dist.unsqueeze(0).to(device),
        )  # [1, 4], already softmaxed

    return dict(zip(EVENT_NAMES, out.squeeze(0).cpu().tolist()))


def predict_tgl(
    tgl_path: str,
    sim_time: Optional[float] = "auto",
    model=None,
    ckpt_path: str = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Predict relative cophylogenetic event frequencies from a .tgl file.

    A well-formatted .tgl bundles the host MSA, symbiont MSA, and the
    host<->symbiont leaf mapping, so this is the single-argument entrypoint.

    tgl_path  : path to the .tgl file.
    sim_time  : "auto" (default) uses Sim_time from the .tgl if present, else the
                model's internal default branch. Pass a float to override, or
                None to force the default branch.
    model     : preloaded Fox model (skip per-call load). Optional.
    ckpt_path, device : forwarded to load_model when model is None.

    Returns {Speciation, HGT, Loss, Duplication} -> float, summing to 1.
    """
    d = read_tgl(tgl_path)
    if sim_time == "auto":
        sim_time = d["sim_time"]
    if model is None:
        model = load_model(ckpt_path, device)
    return _infer(d["host_msa"], d["sym_msa"], d["mapping"], sim_time, model, device)
