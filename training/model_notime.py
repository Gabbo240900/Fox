"""Fox variant trained WITHOUT simulation time.

The no-time checkpoints (e.g. fox_noTime.ckpt) were trained with a model that
had no `sim_time_fc` and whose forward skipped the FiLM conditioning step,
going straight from `feature_mixer` to `event_head`.

Rather than duplicate the full forward, this subclass reuses Fox.forward and
neutralises the FiLM step: Fox applies  x*(1 + tanh(scale)) + shift  where
(scale, shift) = sim_time_fc(sim_time). If sim_time_fc emits zeros, that
collapses to the identity, so the forward is byte-for-byte the no-time variant.

Usage:
    from training.model_notime import FoxNoTime
    model = FoxNoTime.from_checkpoint("fox_noTime.ckpt", device)
"""

import torch
import torch.nn as nn

from training.model import Fox


class _ZeroFiLM(nn.Module):
    """Replacement for sim_time_fc that emits zeros -> FiLM becomes identity."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.out_dim)


class FoxNoTime(Fox):
    """Fox with the sim_time FiLM conditioning disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Drop the time MLP; FiLM is now a no-op regardless of the sim_time arg.
        self.sim_time_fc = _ZeroFiLM(self.concat_dim * 2)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device=None):
        ck  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        h   = ck["hparams"]
        sd  = ck["model"]
        ncl = max(int(k.split(".")[1]) for k in sd
                  if k.startswith("cross_attn_h2p.")) + 1
        m = cls(
            hidden_dim=256, pair_dim=h.get("pair_dim", 64), num_heads=8,
            axial_layers=h.get("axial_layers", 2), use_opm=h.get("use_opm", False),
            use_dist_matrix=True, gradient_checkpointing=False,
            num_cross_layers=ncl, cls_dim=h.get("cls_dim", 512),
            use_flexattention=False, dropout=0.0,
        )
        # no-time checkpoint has no sim_time_fc keys; _ZeroFiLM has no params.
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        if device is not None:
            m = m.to(device)
        return m.eval()
