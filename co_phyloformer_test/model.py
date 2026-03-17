"""
Co-Phyloformer: A transformer-based model for phylogenetic reconciliation event prediction.

Given host and parasite trees (each leaf represented by a protein sequence/MSA),
and a mapping between parasite leaves and host leaves, the model predicts the
frequencies of 4 reconciliation events:
    - Speciation
    - HGT (Horizontal Gene Transfer)
    - Loss
    - Duplication

Architecture overview:
    1. AminoAcidEmbedding: tokenise + embed each AA sequence position → [seq_len, d_model]
    2. LeafAxialAttention: self-attention over sequence positions per leaf → [seq_len, d_model]
       then mean-pool to get a single leaf embedding → [d_model]
    3. TreeEncoder: self-attention over all leaf embeddings + CLS token → [d_model]
    4. CrossAttentionMapper: cross-attend parasite→host per mapping pair, aggregate → [d_model]
    5. MLP head: concatenate the 3 summaries, project to 4 event frequencies (softmax)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Amino acid vocabulary
# ---------------------------------------------------------------------------

# 20 standard amino acids, gap character, and unknown/ambiguous character.
# Index 0 is reserved for padding so that nn.Embedding(padding_idx=0) works.
AA_VOCAB: Dict[str, int] = {
    "<PAD>": 0,  # padding token (not a real AA)
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "-": 21,  # gap
    "X": 22,  # unknown / ambiguous
}
# Any character not in the vocab is mapped to the 'X' (unknown) index.
VOCAB_SIZE = 23  # indices 0–22


def encode_sequence(seq: str) -> torch.LongTensor:
    """Convert an amino acid string to a 1-D LongTensor of token indices.

    Unknown characters are mapped to the 'X' token (index 22).

    Args:
        seq: Amino acid sequence string (e.g. "ACDEFGHIKLM...").

    Returns:
        LongTensor of shape [seq_len].
    """
    unk_idx = AA_VOCAB["X"]
    indices = [AA_VOCAB.get(ch.upper(), unk_idx) for ch in seq]
    return torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------------
# Component 1: AminoAcidEmbedding
# ---------------------------------------------------------------------------


class AminoAcidEmbedding(nn.Module):
    """Embed a tokenised amino acid sequence into a continuous representation.

    Each position in the sequence is independently mapped from its token index
    to a d_model-dimensional vector via a learned embedding table.  A sinusoidal
    positional encoding is then *added* so that attention layers can distinguish
    positions.

    Input:  LongTensor [seq_len]          (token indices, 0 = PAD)
    Output: FloatTensor [seq_len, d_model]
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # padding_idx=0 ensures the PAD embedding stays at zero
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def _positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal positional encoding, shape [seq_len, d_model]."""
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()  # [seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: self.d_model // 2])
        return pe  # [seq_len, d_model]

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            tokens: LongTensor [seq_len] — token indices (0 = PAD).

        Returns:
            FloatTensor [seq_len, d_model].
        """
        seq_len = tokens.size(0)
        x = self.token_embedding(tokens)  # [seq_len, d_model]
        x = x + self._positional_encoding(seq_len, tokens.device)  # [seq_len, d_model]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Component 2: LeafAxialAttention
# ---------------------------------------------------------------------------


class LeafAxialAttention(nn.Module):
    """Per-leaf self-attention over sequence positions.

    Processes ONE leaf at a time.  Applies `num_layers` stacked transformer
    encoder blocks (multi-head self-attention + FFN + LayerNorm + residual)
    over the sequence-position axis.

    Input:  FloatTensor [seq_len, d_model]
    Output: FloatTensor [d_model]   (mean-pooled over seq_len)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # expects [batch, seq, d_model]
            norm_first=True,   # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            FloatTensor [seq_len, d_model].
            padding_mask: BoolTensor  [seq_len] — True for PAD positions
                          (these are ignored in attention and pooling).

        Returns:
            FloatTensor [d_model] — mean-pooled leaf embedding.
        """
        # TransformerEncoder expects batch dimension: [1, seq_len, d_model]
        x = x.unsqueeze(0)  # [1, seq_len, d_model]

        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = padding_mask.unsqueeze(0)  # [1, seq_len]

        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.encoder, x, None, src_key_padding_mask, use_reentrant=False)
        else:
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # x: [1, seq_len, d_model]

        x = x.squeeze(0)  # [seq_len, d_model]

        # Mean-pool over non-PAD positions only
        if padding_mask is not None:
            # mask: True = PAD, so invert for selecting valid positions
            valid = (~padding_mask).float().unsqueeze(-1)  # [seq_len, 1]
            x = (x * valid).sum(0) / valid.sum(0).clamp(min=1.0)  # [d_model]
        else:
            x = x.mean(0)  # [d_model]

        return x


# ---------------------------------------------------------------------------
# Component 3: TreeEncoder
# ---------------------------------------------------------------------------


class TreeEncoder(nn.Module):
    """Encode a set of leaf embeddings into a single tree-level summary vector.

    Steps:
        1. Stack leaf embeddings → [num_leaves, d_model]
        2. Apply self-attention over leaves (num_layers transformer blocks)
        3. Prepend a learnable CLS token and run one additional attention layer
        4. Return the CLS output → [d_model]

    This handles variable numbers of leaves naturally (no padding needed as
    long as each call provides the actual leaves for that tree).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        # Self-attention over leaf embeddings
        leaf_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.leaf_encoder = nn.TransformerEncoder(leaf_layer, num_layers=num_layers)

        # CLS token aggregation layer (attends over [CLS, leaf_1, ..., leaf_N])
        cls_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cls_encoder = nn.TransformerEncoder(cls_layer, num_layers=1)

        # Learnable CLS token (single vector, broadcast over batch)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, leaf_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            leaf_embeddings: list of FloatTensor [d_model], one per leaf.
                             The list length (num_leaves) may vary per sample.

        Returns:
            FloatTensor [d_model] — CLS token output (tree summary).
        """
        # Stack leaves: [1, num_leaves, d_model]  (batch=1 for a single tree)
        x = torch.stack(leaf_embeddings, dim=0).unsqueeze(0)  # [1, num_leaves, d_model]

        # Self-attention over leaves
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.leaf_encoder, x, use_reentrant=False)
        else:
            x = self.leaf_encoder(x)  # [1, num_leaves, d_model]

        # Prepend CLS token: [1, 1 + num_leaves, d_model]
        cls = self.cls_token.expand(1, 1, -1)  # [1, 1, d_model]
        x = torch.cat([cls, x], dim=1)          # [1, 1 + num_leaves, d_model]

        # Attend over CLS + leaves
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.cls_encoder, x, use_reentrant=False)
        else:
            x = self.cls_encoder(x)  # [1, 1 + num_leaves, d_model]

        # Extract CLS output (first position)
        cls_out = x[:, 0, :]  # [1, d_model]
        return cls_out.squeeze(0)  # [d_model]


# ---------------------------------------------------------------------------
# Component 4: CrossAttentionMapper
# ---------------------------------------------------------------------------


class CrossAttentionMapper(nn.Module):
    """Cross-attend from parasite leaf embeddings to host leaf embeddings.

    For each (parasite_leaf → host_leaf) pair provided in the mapping:
        - Query  = parasite leaf embedding [d_model]
        - Key/Value = host leaf embedding  [d_model]
        - Run one cross-attention block
        - Collect the resulting vector [d_model]

    Finally, mean-pool all pair vectors → aggregated representation [d_model].

    This respects the biological mapping: each parasite leaf is explicitly
    related to one (or more) host leaves according to the reconciliation.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Multi-head cross-attention: Q from parasite, K/V from host
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Small FFN after cross-attention (residual block)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        parasite_embeddings: Dict[str, torch.Tensor],
        host_embeddings: Dict[str, torch.Tensor],
        mappings: List[Tuple[str, str]],
    ) -> torch.Tensor:
        """
        Args:
            parasite_embeddings: dict {parasite_leaf_id: FloatTensor [d_model]}
            host_embeddings:     dict {host_leaf_id:     FloatTensor [d_model]}
            mappings:            list of (parasite_leaf_id, host_leaf_id) tuples.
                                 Both IDs must be keys in the respective dicts.

        Returns:
            FloatTensor [d_model] — mean of all cross-attended pair representations.
        """
        pair_outputs: List[torch.Tensor] = []

        for para_id, host_id in mappings:
            # Retrieve embeddings; skip if a leaf is missing (defensive coding)
            if para_id not in parasite_embeddings or host_id not in host_embeddings:
                continue

            # Shape: [1, 1, d_model] — batch=1, seq_len=1 for a single vector
            query = parasite_embeddings[para_id].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
            kv    = host_embeddings[host_id].unsqueeze(0).unsqueeze(0)      # [1, 1, d_model]

            # Cross-attention: parasite query attends to host key/value
            attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
            # attn_out: [1, 1, d_model]

            # Residual + norm (pre-norm pattern)
            attn_out = self.norm(query + self.dropout(attn_out))  # [1, 1, d_model]

            # FFN + residual
            attn_out = self.norm2(attn_out + self.dropout(self.ffn(attn_out)))  # [1, 1, d_model]

            pair_outputs.append(attn_out.squeeze(0).squeeze(0))  # [d_model]

        if len(pair_outputs) == 0:
            # Fallback if no valid mappings: return zeros
            d_model = next(iter(parasite_embeddings.values())).size(0)
            device  = next(iter(parasite_embeddings.values())).device
            return torch.zeros(d_model, device=device)

        # Aggregate over all mapping pairs: mean → [d_model]
        stacked = torch.stack(pair_outputs, dim=0)  # [num_pairs, d_model]
        return stacked.mean(dim=0)                   # [d_model]


# ---------------------------------------------------------------------------
# Component 5: MLP Head
# ---------------------------------------------------------------------------


class MLPHead(nn.Module):
    """Maps concatenated tree summaries to 4 event frequencies.

    Input:  FloatTensor [3 * d_model]   (host_cls ∥ parasite_cls ∥ cross_agg)
    Output: FloatTensor [4]             (probabilities summing to 1.0)

    Architecture:
        Linear(3d, 2d) → LayerNorm → GELU
        Linear(2d, d)  → LayerNorm → GELU
        Linear(d, 4)   → Softmax
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [3 * d_model].

        Returns:
            FloatTensor [4] — event frequencies summing to 1.0.
        """
        logits = self.net(x)            # [4]
        return F.softmax(logits, dim=-1)  # [4]


# ---------------------------------------------------------------------------
# Main Model: CoPhyloformer
# ---------------------------------------------------------------------------


class CoPhyloformer(nn.Module):
    """Co-Phyloformer: transformer model for phylogenetic reconciliation event prediction.

    Given:
        - host_seqs:    dict {leaf_id: amino_acid_sequence_str}
        - parasite_seqs: dict {leaf_id: amino_acid_sequence_str}
        - mappings:     list of (parasite_leaf_id, host_leaf_id) tuples

    Predicts:
        FloatTensor [4] — (Speciation_freq, HGT_freq, Loss_freq, Duplication_freq)
        Values sum to 1.0.

    Full pipeline per forward pass:
        1. Encode + embed each leaf sequence         → [seq_len, d_model]
        2. LeafAxialAttention per leaf               → [d_model]
        3. TreeEncoder for host leaves               → [d_model]  (host CLS)
        4. TreeEncoder for parasite leaves           → [d_model]  (parasite CLS)
        5. CrossAttentionMapper over mapping pairs   → [d_model]
        6. MLPHead on concatenation                  → [4]
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            d_model:                 Model dimensionality (embedding size). Default: 128.
            nhead:                   Number of attention heads. Default: 8.
            num_layers:              Number of transformer encoder layers per module. Default: 2.
            dropout:                 Dropout probability. Default: 0.1.
            gradient_checkpointing:  Trade compute for memory by checkpointing attention layers.
        """
        super().__init__()
        self.d_model = d_model

        # Shared embedding: same vocabulary for host and parasite sequences
        self.embedding = AminoAcidEmbedding(d_model=d_model, dropout=dropout)

        # Shared leaf-level self-attention (weights shared across all leaves of both trees)
        self.leaf_attn = LeafAxialAttention(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Separate tree encoders for host and parasite sides
        self.host_tree_encoder = TreeEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.parasite_tree_encoder = TreeEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Cross-attention over mapping pairs
        self.cross_attn_mapper = CrossAttentionMapper(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Final MLP head
        self.mlp_head = MLPHead(d_model=d_model, dropout=dropout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_leaf(self, seq: str) -> torch.Tensor:
        """Tokenise a sequence string and produce a leaf embedding.

        Steps:
            1. Tokenise → LongTensor [seq_len]
            2. AminoAcidEmbedding → FloatTensor [seq_len, d_model]
            3. LeafAxialAttention + mean-pool → FloatTensor [d_model]

        Args:
            seq: Amino acid sequence string.

        Returns:
            FloatTensor [d_model].
        """
        # Accept either a pre-encoded LongTensor or a raw amino-acid string
        device = next(self.parameters()).device
        if isinstance(seq, torch.Tensor):
            tokens = seq.to(device)
        else:
            tokens = encode_sequence(seq).to(device)  # [seq_len]
        # Padding mask: True for PAD tokens (index 0)
        pad_mask = tokens == 0  # [seq_len]
        # Embed
        x = self.embedding(tokens)           # [seq_len, d_model]
        # Per-leaf self-attention + pool
        leaf_emb = self.leaf_attn(x, pad_mask)  # [d_model]
        return leaf_emb

    def _encode_tree(
        self, seqs: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """Encode all leaves of a tree.

        Args:
            seqs: dict {leaf_id: sequence_string}

        Returns:
            dict {leaf_id: FloatTensor [d_model]}
        """
        return {leaf_id: self._encode_leaf(seq) for leaf_id, seq in seqs.items()}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        host_seqs: Dict[str, str],
        parasite_seqs: Dict[str, str],
        mappings: List[Tuple[str, str]],
    ) -> torch.Tensor:
        """Predict reconciliation event frequencies.

        Args:
            host_seqs:     dict {leaf_id: aa_sequence_str}
                           One entry per host (species) tree leaf.
            parasite_seqs: dict {leaf_id: aa_sequence_str}
                           One entry per parasite (gene) tree leaf.
            mappings:      list of (parasite_leaf_id, host_leaf_id) tuples
                           encoding the leaf-level host–parasite associations.

        Returns:
            FloatTensor [4] — (Speciation_freq, HGT_freq, Loss_freq, Duplication_freq)
            Values are non-negative and sum to 1.0.
        """
        # ----------------------------------------------------------------
        # Step 1 & 2: Embed + per-leaf attention for all leaves
        # host_leaf_embeddings:     {leaf_id: [d_model]}
        # parasite_leaf_embeddings: {leaf_id: [d_model]}
        # ----------------------------------------------------------------
        host_leaf_embeddings     = self._encode_tree(host_seqs)      # {id: [d_model]}
        parasite_leaf_embeddings = self._encode_tree(parasite_seqs)  # {id: [d_model]}

        # ----------------------------------------------------------------
        # Step 3: Tree-level encoding with CLS token
        # host_cls:     [d_model]
        # parasite_cls: [d_model]
        # ----------------------------------------------------------------
        host_cls     = self.host_tree_encoder(list(host_leaf_embeddings.values()))      # [d_model]
        parasite_cls = self.parasite_tree_encoder(list(parasite_leaf_embeddings.values()))  # [d_model]

        # ----------------------------------------------------------------
        # Step 4: Cross-attention over mapping pairs
        # cross_agg: [d_model]
        # ----------------------------------------------------------------
        cross_agg = self.cross_attn_mapper(
            parasite_embeddings=parasite_leaf_embeddings,
            host_embeddings=host_leaf_embeddings,
            mappings=mappings,
        )  # [d_model]

        # ----------------------------------------------------------------
        # Step 5: MLP head
        # Concatenate the three d_model summaries → [3 * d_model]
        # ----------------------------------------------------------------
        combined = torch.cat([host_cls, parasite_cls, cross_agg], dim=-1)  # [3 * d_model]
        freqs = self.mlp_head(combined)  # [4]

        return freqs


# ---------------------------------------------------------------------------
# Batched forward utility
# ---------------------------------------------------------------------------


def forward_batch(
    model: CoPhyloformer,
    batch: List[
        Tuple[
            Dict[str, str],          # host_seqs
            Dict[str, str],          # parasite_seqs
            List[Tuple[str, str]],   # mappings
        ]
    ],
) -> torch.Tensor:
    """Run CoPhyloformer over a list of samples and return stacked predictions.

    Because trees have variable numbers of leaves and variable sequence lengths,
    true tensor-level batching would require complex dynamic padding.  Instead
    this utility loops over samples and stacks the resulting [4] vectors.  For
    training at scale, wrapping each sample's tensors in a DataLoader and
    relying on this function (or a collate_fn) is the recommended approach.

    Args:
        model: An instantiated CoPhyloformer (in eval or train mode).
        batch: List of (host_seqs, parasite_seqs, mappings) tuples.

    Returns:
        FloatTensor [batch_size, 4] — one frequency vector per sample.
    """
    outputs = [model(host_seqs, parasite_seqs, mappings) for host_seqs, parasite_seqs, mappings in batch]
    return torch.stack(outputs, dim=0)  # [batch_size, 4]


# ---------------------------------------------------------------------------
# Quick sanity-check (runs when the script is executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import string

    def _random_seq(length: int = 50) -> str:
        aas = list("ACDEFGHIKLMNPQRSTVWY")
        return "".join(random.choices(aas, k=length))

    torch.manual_seed(42)

    # Instantiate model with defaults
    model = CoPhyloformer(d_model=128, nhead=8, num_layers=2, dropout=0.1)
    model.eval()

    # Build a dummy sample
    n_host     = 5
    n_parasite = 4
    host_seqs     = {f"H{i}": _random_seq(60) for i in range(n_host)}
    parasite_seqs = {f"P{i}": _random_seq(45) for i in range(n_parasite)}
    # Mappings: each parasite leaf → one or more host leaves
    mappings = [("P0", "H0"), ("P0", "H2"), ("P1", "H1"), ("P2", "H3"), ("P3", "H4")]

    with torch.no_grad():
        out = model(host_seqs, parasite_seqs, mappings)

    print("Output shape:", out.shape)           # Expected: torch.Size([4])
    print("Output values:", out)
    print("Sum:", out.sum().item())              # Expected: ~1.0

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
