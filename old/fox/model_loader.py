"""Load the pretrained Fox checkpoint into an eval-ready model."""

import os
import sys
from pathlib import Path
from functools import lru_cache

import torch

# Repo root holds both training/ (model code) and fox.ckpt (weights).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_ckpt() -> str:
    """Resolve the checkpoint path: $FOX_CKPT, then repo-root, then cwd."""
    env = os.environ.get("FOX_CKPT")
    if env:
        return env
    for cand in (_REPO_ROOT / "fox.ckpt", Path.cwd() / "fox.ckpt"):
        if cand.exists():
            return str(cand)
    return str(_REPO_ROOT / "fox.ckpt")  # default for the error message


def _import_fox_class():
    # training/ is not (yet) an installed package; make it importable.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from training.model import Fox  # noqa: E402
    return Fox


@lru_cache(maxsize=2)
def load_model(ckpt_path: str = None, device: str = "cpu"):
    """Build a Fox model from a checkpoint and return it in eval mode.

    ckpt_path : path to fox.ckpt. Defaults to $FOX_CKPT or the repo-root file.
    device    : "cpu", "cuda", or "mps".

    Cached per (ckpt_path, device) so repeated calls are cheap.
    """
    ckpt_path = ckpt_path or _default_ckpt()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path!r}. "
            "Set $FOX_CKPT or pass ckpt_path=."
        )

    Fox = _import_fox_class()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})

    # num_cross_layers is not stored in hparams; infer it from the state dict.
    num_cross_layers = max(
        int(k.split(".")[1]) for k in ckpt["model"] if k.startswith("cross_attn_h2p.")
    ) + 1

    model = Fox(
        hidden_dim=256,
        pair_dim=hp.get("pair_dim", 64),
        num_heads=8,
        axial_layers=hp.get("axial_layers", 2),
        use_opm=hp.get("use_opm", False),
        use_dist_matrix=True,
        gradient_checkpointing=False,
        num_cross_layers=num_cross_layers,
        cls_dim=hp.get("cls_dim", 512),
        use_flexattention=False,  # FlexAttention needs CUDA; off for portable inference
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model
