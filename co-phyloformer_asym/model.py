
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSAEmbedder(nn.Module):
    """
    Phyloformer-2-style MSA embedder adapted for Co-phyloformer's
    channels-last [B, N, S, D] layout and integer-token input.

    Equivalent to pf_sdk's Conv2d(23, embed_dim, 1) + ReLU, but operating
    on integer IDs that are one-hot-encoded on the fly.
    """

    VOCAB_SIZE = 24  # 0-19 AAs, 20=gap, 21=UNK, 22=PAD, 23=MASK

    def __init__(self, seq_dim: int) -> None:
        super().__init__()
        # bias=False mirrors Conv2d with no bias; kaiming_uniform_ default init
        self.proj = nn.Linear(self.VOCAB_SIZE, seq_dim, bias=False)
        self.act  = nn.ReLU()

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids:   [B, N, S]  integer token ids
        Returns: [B, N, S, seq_dim]
        """
        # One-hot encode: [B, N, S] → [B, N, S, VOCAB_SIZE]
        x_oh = F.one_hot(x_ids.long(), num_classes=self.VOCAB_SIZE).to(
            dtype=self.proj.weight.dtype
        )
        # Project + activate: [B, N, S, seq_dim]
        return self.act(self.proj(x_oh))


# ─────────────────────────────────────────────────────────────────────────────
# EvoPF-style MSA + Pair blocks
# Adapted from Phyloformer-2 (BayesNJ/EvoPF) to the Co-phyloformer
# channels-last [B, N, S, D] tensor layout.
#
# Architecture per EvoPFBlockLite:
#   MSA  track : ColAttnPairBias → GatedRowAttention → MSA FFN
#   Pair track : ConcatPairUpdate → PairAttnBlock    → Pair FFN
# ─────────────────────────────────────────────────────────────────────────────

class GatedRowAttention(nn.Module):
    """
    Row (sequence-axis) attention: each leaf independently attends along its
    own residue axis S.  Pre-norm + sigmoid gating (Phyloformer-2 / Evoformer).

    Input / output: [B, N, S, D]   (residual added by caller)
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.mha      = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                               dropout=dropout, bias=False)
        self.g_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        x:        [B, N, S, D]
        pad_mask: [B, N, S]   True = PAD token (ignore)
        """
        B, N, S, D = x.shape
        xn = self.norm(x).view(B * N, S, D)
        pm = pad_mask.view(B * N, S)

        g = torch.sigmoid(self.g_proj(xn))          # [B*N, S, D]

        # NaN guard: fully-PAD leaf rows would produce NaN in softmax
        all_pad = pm.all(dim=1)                      # [B*N]
        if all_pad.any():
            xn = xn.clone(); pm = pm.clone()
            xn[all_pad] = 0.0
            pm[all_pad] = False

        attn_out, _ = self.mha(xn, xn, xn, key_padding_mask=pm, need_weights=False)
        out = self.out_proj(g * attn_out)

        if all_pad.any():
            out[all_pad] = 0.0

        return out.view(B, N, S, D)


class ColAttnPairBias(nn.Module):
    """
    Column (leaf-axis) attention: leaves attend to each other per residue
    column, biased by an explicit pair representation and sigmoid-gated.
    Follows Phyloformer-2's ColAttenPairBias, adapted to [B, N, S, D].

    The pair bias [B, N, N, H] is broadcast over S without materialising the
    full [B, S, H, N, N] tensor, keeping peak memory to O(B·S·H·N²) for the
    attention scores only.

    Input / output MSA:  [B, N, S, D]   (residual added by caller)
    Input pair:          [B, N, N, pair_dim]
    """

    def __init__(self, dim: int, pair_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.msa_norm  = nn.LayerNorm(dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.g_proj   = nn.Linear(dim, dim)                      # gating
        self.b_proj   = nn.Linear(pair_dim, n_heads, bias=False) # pair → per-head bias
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, S, D]
        pairs: torch.Tensor,        # [B, N, N, pair_dim]
        leaf_pad: torch.Tensor,     # [B, N]  True = leaf is fully PAD
    ) -> torch.Tensor:
        B, N, S, D = x.shape
        H, d_h = self.n_heads, self.head_dim

        xn = self.msa_norm(x)                                    # [B, N, S, D]
        b  = self.b_proj(self.pair_norm(pairs))                  # [B, N, N, H]

        # Reshape to process all residue columns in one batch: [B*S, N, D]
        xl = xn.permute(0, 2, 1, 3).contiguous().view(B * S, N, D)

        def _proj(linear):
            return linear(xl).view(B * S, N, H, d_h).transpose(1, 2)  # [B*S, H, N, d_h]

        q = _proj(self.q_proj)
        k = _proj(self.k_proj)
        v = _proj(self.v_proj)
        g = torch.sigmoid(_proj(self.g_proj))

        # Attention scores [B*S, H, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add pair bias [B, H, N, N] → broadcast over S (no extra allocation)
        bias = b.permute(0, 3, 1, 2)                             # [B, H, N, N]
        scores = scores.view(B, S, H, N, N) + bias.unsqueeze(1) # [B, S, H, N, N]

        # Key-padding mask for PAD leaves — broadcast over S without expand
        if leaf_pad.any():
            # [B, 1, 1, 1, N] True = ignore that key leaf
            km = leaf_pad[:, None, None, None, :]
            scores = scores.masked_fill(km, -1e9)

            # NaN guard: batch items where ALL leaves are PAD
            all_leaves_pad = leaf_pad.all(dim=1)          # [B]
            if all_leaves_pad.any():
                scores[all_leaves_pad, :, :, :, 0] = 0.0  # unmask one key

        scores = scores.view(B * S, H, N, N)
        attn   = self.dropout(torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype))

        attn_out = g * torch.matmul(attn, v)              # [B*S, H, N, d_h]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B * S, N, D)
        out      = self.out_proj(attn_out)                # [B*S, N, D]

        # Back to [B, N, S, D]
        return out.view(B, S, N, D).permute(0, 2, 1, 3).contiguous()


class ConcatPairUpdate(nn.Module):
    """
    Update the pair representation from the MSA via mean-pooling + concatenation.
    Memory-efficient alternative to Outer Product Mean (same idea, lower cost).
    Adapted from EvoPF's ConcatenationPairUpdate.

    Input MSA:    [B, N, S, D]
    Output pairs: [B, N, N, pair_dim]
    """

    def __init__(self, dim: int, pair_dim: int, inner_dim: int = 32):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.a_proj = nn.Linear(dim, inner_dim)
        self.b_proj = nn.Linear(dim, inner_dim)
        self.o_mlp  = nn.Sequential(
            nn.Linear(2 * inner_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, S, D]
        xn = self.norm(x)
        # Mean-pool over sequence positions S → [B, N, inner_dim]
        a = self.a_proj(xn).mean(dim=2)           # [B, N, inner_dim]
        b = self.b_proj(xn).mean(dim=2)           # [B, N, inner_dim]
        # Pairwise: broadcast to [B, N, N, inner_dim]
        a = a.unsqueeze(2)                        # [B, N, 1, inner_dim]
        b = b.unsqueeze(1)                        # [B, 1, N, inner_dim]
        c = torch.cat([a + b, (a - b).abs()], dim=-1)   # [B, N, N, 2*inner_dim]
        return self.o_mlp(c)                      # [B, N, N, pair_dim]


class OuterProductMean(nn.Module):
    """
    Update pair representation via Outer Product Mean (AlphaFold2 / Phyloformer-2).
    For every residue column s, computes the outer product a[:,i,s,:] ⊗ b[:,j,s,:]
    between all leaf pairs, then averages over S.  Captures per-position co-variation
    between leaves — more expressive than ConcatPairUpdate but O(N²·inner_dim²) memory.

    Safe memory budget (inner_dim=16, leaf_attn_max_leaves guard active):
      N=50,  B=8 → ~50 MB   ✓
      N=128, B=8 → ~335 MB  (pair track skipped by guard at that scale)

    Input MSA:    [B, N, S, D]
    Output pairs: [B, N, N, pair_dim]
    """

    def __init__(self, dim: int, pair_dim: int, inner_dim: int = 16):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.a_proj = nn.Linear(dim, inner_dim)
        self.b_proj = nn.Linear(dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim * inner_dim, pair_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, S, D]
        xn = self.norm(x)
        a = self.a_proj(xn)   # [B, N, S, inner_dim]
        b = self.b_proj(xn)   # [B, N, S, inner_dim]
        # Outer product averaged over S: einsum contracts S, outer on inner_dim
        # Result: [B, N, N, inner_dim, inner_dim]
        outer = torch.einsum('bnsd,bNsD->bnNdD', a, b) / x.shape[2]
        B, N, _, d, _ = outer.shape
        o = outer.reshape(B, N, N, d * d)   # [B, N, N, inner_dim²]
        return self.o_proj(o)               # [B, N, N, pair_dim]


class PairAttnBlock(nn.Module):
    """
    Gated attention on the pair track, applied row-wise (per source leaf).
    Follows Phyloformer-2's PairAttention (asymmetric mode).

    Input / output: [B, N, N, pair_dim]   (residual added by caller)
    """

    def __init__(self, pair_dim: int, n_heads: int):
        super().__init__()
        self.norm     = nn.LayerNorm(pair_dim)
        self.mha      = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True, bias=False)
        self.g_proj   = nn.Linear(pair_dim, pair_dim)
        self.out_proj = nn.Linear(pair_dim, pair_dim)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: [B, N, N, pair_dim]
        B, N, _, D = pairs.shape
        r  = self.norm(pairs).view(B * N, N, D)      # [B*N, N, pair_dim]
        g  = torch.sigmoid(self.g_proj(r))
        out, _ = self.mha(r, r, r, need_weights=False)
        return self.out_proj(g * out).view(B, N, N, D)


class EvoPFBlockLite(nn.Module):
    """
    Full Phyloformer-2 EvoPF block adapted for Co-phyloformer's [B, N, S, D] layout.

    MSA  track (order mirrors EvoPFBlock):
        1. ColAttnPairBias  — leaf attention biased by pair repr, gated
        2. GatedRowAttention — residue attention, gated
        3. MSA FFN (pre-norm, GELU)

    Pair track (interleaved with MSA track, only when N ≤ leaf_attn_max_leaves):
        4. ConcatPairUpdate  — derive pair signal from updated MSA
        5. PairAttnBlock     — refine pair repr with gated attention
        6. Pair FFN (pre-norm, GELU)

    When N > leaf_attn_max_leaves the column attention and pair track are
    skipped (row attention + FFN still run), identical to the original guard.
    """

    def __init__(
        self,
        seq_dim: int,
        n_heads: int,
        pair_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        leaf_attn_max_leaves: int = 128,
        use_opm: bool = False,
    ):
        super().__init__()
        self.leaf_attn_max_leaves = int(leaf_attn_max_leaves)

        # ── MSA track ────────────────────────────────────────────────────────
        self.col_attn = ColAttnPairBias(seq_dim, pair_dim, n_heads, dropout)
        self.row_attn = GatedRowAttention(seq_dim, n_heads, dropout)

        self.msa_norm = nn.LayerNorm(seq_dim)
        self.msa_ff   = nn.Sequential(
            nn.Linear(seq_dim, seq_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim * ff_mult, seq_dim),
        )

        # ── Pair track (only active when N ≤ leaf_attn_max_leaves) ───────────
        # use_opm=True  → OuterProductMean (more expressive, higher memory)
        # use_opm=False → ConcatPairUpdate (memory-efficient default)
        self.pair_update = (
            OuterProductMean(seq_dim, pair_dim, inner_dim=16)
            if use_opm else
            ConcatPairUpdate(seq_dim, pair_dim, inner_dim=32)
        )
        self.pair_attn   = PairAttnBlock(pair_dim, n_heads)

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_ff   = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * ff_mult, pair_dim),
        )

    def forward(
        self,
        x: torch.Tensor,      # [B, N, S, D]
        pairs: torch.Tensor,  # [B, N, N, pair_dim]
        x_ids: torch.Tensor,  # [B, N, S]
    ):
        B, N, S, D = x.shape
        pad_mask  = (x_ids == 22)               # [B, N, S]  True = PAD residue
        leaf_pad  = ~(x_ids != 22).any(dim=2)  # [B, N]     True = fully-PAD leaf

        do_leaf = (self.leaf_attn_max_leaves <= 0 or N <= self.leaf_attn_max_leaves)

        # ── MSA track ────────────────────────────────────────────────────────
        if do_leaf:
            x = x + self.col_attn(x, pairs, leaf_pad)  # 1. col attention w/ pair bias

        x = x + self.row_attn(x, pad_mask)             # 2. row attention (always)
        x = x + self.msa_ff(self.msa_norm(x))          # 3. MSA FFN

        # ── Pair track ───────────────────────────────────────────────────────
        if do_leaf:
            pairs = pairs + self.pair_update(x)             # 4. update from MSA
            pairs = pairs + self.pair_attn(pairs)           # 5. pair attention
            pairs = pairs + self.pair_ff(self.pair_norm(pairs))  # 6. pair FFN

        return x, pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLS-level leaf transformer (unchanged from original Co-phyloformer)
# ─────────────────────────────────────────────────────────────────────────────

class FlashMSAEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        self.qkv  = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x:               (B, N, D)
        key_padding_mask:(B, N)  True = PAD/invalid token
        """
        B, N, D = x.shape
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()

        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)

        if key_padding_mask is not None:
            km = key_padding_mask[:, None, None, :]
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                km = km.clone()
                km[all_masked, :, :, 0] = False
            scores = scores.masked_fill(km, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v_fp32)

        out = out.to(dtype=q.dtype)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MSA Encoder
# ─────────────────────────────────────────────────────────────────────────────

class MSAEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        seq_dim=128,
        pair_dim=32,
        num_layers=4,
        num_heads=8,
        axial_layers=2,
        leaf_attn_max_leaves=128,
        gradient_checkpointing: bool = False,
        use_opm: bool = False,
        use_dist_matrix: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.pair_dim        = pair_dim
        self.use_dist_matrix = use_dist_matrix

        # seq_heads: scaled proportionally so head_dim stays constant
        seq_heads = max(1, num_heads * seq_dim // hidden_dim)

        self.embedder = MSAEmbedder(seq_dim)

        # Optional: project scalar JC distances into pair_dim space to
        # initialise the pair track with real phylogenetic signal.
        if use_dist_matrix:
            self.dist_proj = nn.Sequential(
                nn.Linear(1, pair_dim),
                nn.GELU(),
                nn.Linear(pair_dim, pair_dim),
            )

        # EvoPF-style axial blocks (MSA + pair track)
        self.n_evopf_layers = int(axial_layers)
        self.evopf_blocks   = nn.ModuleList([
            EvoPFBlockLite(
                seq_dim, seq_heads, pair_dim,
                ff_mult=4, dropout=0.1,
                leaf_attn_max_leaves=leaf_attn_max_leaves,
                use_opm=use_opm,
            )
            for _ in range(self.n_evopf_layers)
        ])

        self.norm      = nn.LayerNorm(seq_dim)
        self.leaf_proj = nn.Linear(seq_dim, hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            FlashMSAEncoderLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm    = nn.LayerNorm(hidden_dim)
        self.mask_token_id = 23

    def forward(self, x, dist_matrix: Optional[torch.Tensor] = None):
        """
        x:           (B, N, S)  token ids
        dist_matrix: (B, N, N)  optional Jukes-Cantor pairwise distances.
                     When provided and use_dist_matrix=True, used to initialise
                     the pair track with real phylogenetic signal instead of zeros.
        """
        x_ids = x
        B, N, S = x_ids.shape

        # Phyloformer-2-style embedding: one-hot → Linear + ReLU, no PE
        x = self.embedder(x_ids)  # (B, N, S, seq_dim)

        # Initialise pair representation
        if self.use_dist_matrix and dist_matrix is not None:
            # Project scalar distances → [B, N, N, pair_dim]
            pairs = self.dist_proj(dist_matrix.unsqueeze(-1).to(x.dtype))
        else:
            # Zeros: ConcatPairUpdate / OPM will populate from the first block
            pairs = torch.zeros(B, N, N, self.pair_dim, device=x.device, dtype=x.dtype)

        # EvoPF blocks (MSA track + pair track)
        for blk in self.evopf_blocks:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x, pairs = checkpoint(blk, x, pairs, x_ids, use_reentrant=False)
            else:
                x, pairs = blk(x, pairs, x_ids)

        # Max-pool over S, ignoring PAD tokens
        pad_mask = (x_ids != 22)                           # (B, N, S)
        x_masked = x.masked_fill(~pad_mask.unsqueeze(-1), -1e4)
        x, _     = torch.max(x_masked, dim=2)             # (B, N, seq_dim)

        x = self.norm(x)
        x = self.leaf_proj(x)                             # (B, N, hidden_dim)

        cls_token = self.cls_token.expand(B, -1, -1)      # (B, 1, hidden_dim)
        x = torch.cat([cls_token, x], dim=1)              # (B, N+1, hidden_dim)

        # CLS-level key-padding mask
        leaf_present    = (x_ids != 22).any(dim=2)        # (B, N)
        cls_present     = torch.ones((B, 1), device=x.device, dtype=torch.bool)
        seq_present     = torch.cat([cls_present, leaf_present], dim=1)  # (B, N+1)
        key_padding_mask = ~seq_present                   # (B, N+1)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(
                    lambda _x: layer(_x, key_padding_mask=key_padding_mask),
                    x, use_reentrant=False,
                )
            else:
                x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return x, x[:, 0]   # full leaf embeddings + CLS


# ─────────────────────────────────────────────────────────────────────────────
# Co-phyloformer  (cross-attention part unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Cophyloformer(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        seq_dim=128,
        pair_dim=32,
        num_layers=4,
        num_heads=8,
        axial_layers=2,
        gradient_checkpointing: bool = False,
        use_opm: bool = False,
        use_dist_matrix: bool = False,
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.seq_dim         = seq_dim
        self.pair_dim        = pair_dim
        self.num_layers      = num_layers
        self.num_heads       = num_heads
        self.embedding_dim   = hidden_dim
        self.use_dist_matrix = use_dist_matrix

        self.host_encoder = MSAEncoder(
            hidden_dim, seq_dim, pair_dim, num_layers, num_heads,
            axial_layers=axial_layers,
            gradient_checkpointing=gradient_checkpointing,
            use_opm=use_opm,
            use_dist_matrix=use_dist_matrix,
        )
        self.parasite_encoder = MSAEncoder(
            hidden_dim, seq_dim, pair_dim, num_layers, num_heads,
            axial_layers=axial_layers,
            gradient_checkpointing=gradient_checkpointing,
            use_opm=use_opm,
            use_dist_matrix=use_dist_matrix,
        )

        # ── Cross-attention (Co-phyloformer specific, no equivalent in Phyloformer-2) ──
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )

        self.num_cross_layers = 2
        self.cross_attn_h2p = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
            for _ in range(self.num_cross_layers)
        ])
        self.cross_attn_p2h = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
            for _ in range(self.num_cross_layers)
        ])
        self.cross_norms_h = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_cross_layers)])
        self.cross_norms_p = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_cross_layers)])
        self.cross_ffns_h  = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            for _ in range(self.num_cross_layers)
        ])
        self.cross_ffns_p  = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            for _ in range(self.num_cross_layers)
        ])

        self.pair_pool_score_h = nn.Linear(hidden_dim, 1)
        self.pair_pool_score_p = nn.Linear(hidden_dim, 1)

        self.concat_dim = 4 * hidden_dim

        self.sim_time_fc = nn.Sequential(
            nn.Linear(1, self.concat_dim),
            nn.GELU(),
            nn.Linear(self.concat_dim, self.concat_dim * 2),
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.GELU(),
            nn.LayerNorm(self.concat_dim),
            nn.Dropout(0.1),
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.GELU(),
        )

        self.event_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, host_msa, parasite_msa, mappings, sim_time,
                host_dist: Optional[torch.Tensor] = None,
                para_dist: Optional[torch.Tensor] = None):
        """
        host_dist / para_dist: (B, N, N) Jukes-Cantor distance matrices.
        Only used when the encoder was built with use_dist_matrix=True.
        """
        host_emb,     host_cls     = self.host_encoder(host_msa, dist_matrix=host_dist)
        parasite_emb, parasite_cls = self.parasite_encoder(parasite_msa, dist_matrix=para_dist)

        global_cross, _ = self.cross_attention(
            host_cls.unsqueeze(1),
            parasite_cls.unsqueeze(1),
            parasite_cls.unsqueeze(1),
        )
        global_cross = global_cross.squeeze(1)

        batch_size = host_msa.shape[0]
        hidden_dim = host_emb.shape[-1]
        max_pairs  = max((len(m) for m in mappings), default=0)
        L          = max_pairs + 1   # +1 for CLS

        device = host_emb.device

        host_idx = torch.full((batch_size, max_pairs), -1, device=device, dtype=torch.long)
        para_idx = torch.full((batch_size, max_pairs), -1, device=device, dtype=torch.long)

        for i, mapping in enumerate(mappings):
            if len(mapping) == 0:
                continue
            h = torch.as_tensor([hp[0] for hp in mapping], device=device, dtype=torch.long) + 1
            p = torch.as_tensor([hp[1] for hp in mapping], device=device, dtype=torch.long) + 1
            h = h.clamp(min=0, max=host_emb.shape[1] - 1)
            p = p.clamp(min=0, max=parasite_emb.shape[1] - 1)
            host_idx[i, :h.numel()] = h
            para_idx[i, :p.numel()] = p

        safe_host_idx = host_idx.clamp(min=0)
        safe_para_idx = para_idx.clamp(min=0)

        host_gather = host_emb.gather(1, safe_host_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))
        para_gather = parasite_emb.gather(1, safe_para_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))

        host_gather = host_gather.masked_fill((host_idx == -1).unsqueeze(-1), 0)
        para_gather = para_gather.masked_fill((para_idx == -1).unsqueeze(-1), 0)

        host_seq     = torch.cat([host_emb[:, 0:1, :],     host_gather], dim=1)  # (B, L, D)
        parasite_seq = torch.cat([parasite_emb[:, 0:1, :], para_gather], dim=1)

        pair_present = torch.ones((batch_size, L), device=device, dtype=torch.bool)
        if max_pairs > 0:
            pair_present[:, 1:] = (host_idx != -1) & (para_idx != -1)
        kv_pad_mask = ~pair_present

        cross_host = host_seq
        cross_para = parasite_seq

        for i in range(self.num_cross_layers):
            h_attn, _ = self.cross_attn_h2p[i](self.cross_norms_h[i](cross_host), cross_para, cross_para, key_padding_mask=kv_pad_mask)
            cross_host = cross_host + h_attn
            cross_host = cross_host + self.cross_ffns_h[i](self.cross_norms_h[i](cross_host))

            p_attn, _ = self.cross_attn_p2h[i](self.cross_norms_p[i](cross_para), cross_host, cross_host, key_padding_mask=kv_pad_mask)
            cross_para = cross_para + p_attn
            cross_para = cross_para + self.cross_ffns_p[i](self.cross_norms_p[i](cross_para))

        def masked_softmax_pool(logits, seq, mask):
            logits = logits.masked_fill(~mask, -1e4)
            w = F.softmax(logits.float(), dim=1).to(dtype=seq.dtype)
            w = w * mask.to(dtype=seq.dtype)
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return torch.sum(seq * w.unsqueeze(-1), dim=1)

        host_pooled  = masked_softmax_pool(self.pair_pool_score_h(cross_host).squeeze(-1), cross_host, pair_present)
        para_pooled  = masked_softmax_pool(self.pair_pool_score_p(cross_para).squeeze(-1), cross_para, pair_present)

        attended_pairs = torch.cat([host_cls, parasite_cls, host_pooled, para_pooled], dim=-1)
        attended_pairs = self.feature_mixer(attended_pairs)

        if sim_time is not None:
            gamma_beta = self.sim_time_fc(sim_time)
            scale, shift = gamma_beta.chunk(2, dim=-1)
            scale = torch.tanh(scale)
            attended_pairs = attended_pairs * (1 + scale) + shift

        logits = self.event_head(attended_pairs)
        return torch.sigmoid(logits)
