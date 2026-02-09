import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialMSABlockLite(nn.Module):
    """Lightweight axial attention over a single MSA (B, N, S, D).

    - Residue attention: along S, per leaf (B*N, S, D)
    - Leaf attention: along N, per residue column (B*S, N, D)

    Leaf-attention is skipped when N > leaf_attn_max_leaves to stay lightweight.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0, leaf_attn_max_leaves: int = 128):
        super().__init__()
        self.leaf_attn_max_leaves = int(leaf_attn_max_leaves)

        self.res_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.leaf_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.norm_r1 = nn.LayerNorm(hidden_dim)
        self.norm_r2 = nn.LayerNorm(hidden_dim)
        self.norm_l1 = nn.LayerNorm(hidden_dim)
        self.norm_l2 = nn.LayerNorm(hidden_dim)

        self.ff_r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.ff_l = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, N, S, D), x_ids: (B, N, S)
        B, N, S, D = x.shape

        # Residue attention (along S) per leaf
        xr = self.norm_r1(x).view(B * N, S, D)
        pad_mask_r = (x_ids.view(B * N, S) == 22)  # True where PAD

        # If an entire row is PAD, MultiheadAttention can produce NaNs.
        # We neutralize those rows by (a) zeroing inputs, (b) disabling the mask for that row,
        # and (c) zeroing outputs afterward.
        all_pad_r = pad_mask_r.all(dim=1)  # (B*N,)
        if all_pad_r.any():
            xr = xr.clone()
            pad_mask_r = pad_mask_r.clone()
            xr[all_pad_r] = 0
            pad_mask_r[all_pad_r] = False

        attn_r, _ = self.res_attn(xr, xr, xr, key_padding_mask=pad_mask_r, need_weights=False)
        xr = xr + attn_r
        xr = xr + self.ff_r(self.norm_r2(xr))

        if all_pad_r.any():
            xr[all_pad_r] = 0

        x = xr.view(B, N, S, D)

        # Leaf attention (along N) per residue column (skip for large N)
        if self.leaf_attn_max_leaves > 0 and N <= self.leaf_attn_max_leaves:
            xl = self.norm_l1(x)
            # (B, N, S, D) -> (B, S, N, D) -> (B*S, N, D)
            xl = xl.permute(0, 2, 1, 3).contiguous().view(B * S, N, D)

            # Mask padded leaves (leaf is padded if all residues are PAD)
            leaf_present = (x_ids != 22).any(dim=2)  # (B, N)
            pad_mask_l = (~leaf_present).unsqueeze(1).expand(B, S, N).contiguous().view(B * S, N)

            # Same NaN guard for fully-masked rows
            all_pad_l = pad_mask_l.all(dim=1)  # (B*S,)
            if all_pad_l.any():
                xl = xl.clone()
                pad_mask_l = pad_mask_l.clone()
                xl[all_pad_l] = 0
                pad_mask_l[all_pad_l] = False

            attn_l, _ = self.leaf_attn(xl, xl, xl, key_padding_mask=pad_mask_l, need_weights=False)
            xl = xl + attn_l
            xl = xl + self.ff_l(self.norm_l2(xl))

            if all_pad_l.any():
                xl[all_pad_l] = 0

            # Back to (B, N, S, D)
            x = xl.view(B, S, N, D).permute(0, 2, 1, 3).contiguous()

        return x

# Understand model size where it comes from parameters and bottlenecks for memory 
class FlashMSAEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, N, D)

        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x
    
class MSAEncoder(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=8, num_heads=8, axial_layers=1, leaf_attn_max_leaves=128):# increase embedding and layers reduce batch size 
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)
        # MSA-Transformer-style axial attention blocks BEFORE pooling
        self.axial_layers = int(axial_layers)
        self.axial_blocks = nn.ModuleList([
            AxialMSABlockLite(hidden_dim, num_heads, ff_mult=4, dropout=0.0, leaf_attn_max_leaves=leaf_attn_max_leaves)
            for _ in range(self.axial_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool_weights = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            FlashMSAEncoderLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        #self.dropout = nn.Dropout(0.1)
        self.mask_token_id = 23
        

    def forward(self, x):
        # x: (B, N, S) token ids
        x_ids = x
        x = self.embedding(x_ids)  # (B, N, S, D)

        # Axial attention over the MSA grid (N x S) before pooling
        for blk in self.axial_blocks:
            x = blk(x, x_ids)

        # Mask padding positions (PAD token id = 22) so pooling ignores padded tokens
        pad_mask = (x_ids != 22)  # (B, N, S)
        logits = self.pool_weights(x).squeeze(-1)  # (B, N, S)
        logits = logits.masked_fill(~pad_mask, -1e4)

        # softmax in fp32 + renorm prevents NaNs when an entire leaf is PAD
        weights = F.softmax(logits.float(), dim=2).to(dtype=x.dtype)
        weights = weights * pad_mask.to(dtype=x.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        weights = weights / denom

        x = torch.sum(x * weights.unsqueeze(-1), dim=2)  # (B, N, D)
        x = self.norm(x)
        #x = self.dropout(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x, x[:, 0]

class Cophyloformer(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=8, num_heads=8):
        super(Cophyloformer, self).__init__()
        # Store hyperparameters for W&B logging
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        #self.dropout = 0.1
        self.embedding_dim = hidden_dim
        
        self.host_encoder = MSAEncoder(hidden_dim, num_layers, num_heads)
        self.parasite_encoder = MSAEncoder(hidden_dim, num_layers, num_heads)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Learnable pooling over (CLS + mapped pair tokens)
        self.pair_pool_score = nn.Linear(hidden_dim, 1) 

        #--- sim_time modulation disabled ---
        self.sim_time_fc = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.Identity()
        )
        self.concat_dim = 3 * hidden_dim
        self.cospeciation_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, hidden_dim),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.switch_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, hidden_dim),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, host_msa, parasite_msa, mappings, sim_time):
        # Encode host and parasite MSAs

        host_emb, host_cls = self.host_encoder(host_msa)  # (batch, num_leaves+1, hidden_dim), (batch, hidden_dim)
        parasite_emb, parasite_cls = self.parasite_encoder(parasite_msa)  # (batch, num_leaves+1, hidden_dim), (batch, hidden_dim)

        batch_size = host_msa.shape[0]
        hidden_dim = host_emb.shape[-1]
        mapped_pair_features = torch.zeros(batch_size, hidden_dim * 3, device=host_msa.device)

        for i, mapping in enumerate(mappings):
            # Always include CLS↔CLS as a global pair
            host_nodes = [host_emb[i, 0]]
            parasite_nodes = [parasite_emb[i, 0]]

            # Add leaf↔leaf pairs from mappings (leaf indices start at 0, but embeddings have CLS at 0)
            for h_idx, p_idx in mapping:
                h = h_idx + 1
                p = p_idx + 1
                if 0 <= h < host_emb.shape[1] and 0 <= p < parasite_emb.shape[1]:
                    host_nodes.append(host_emb[i, h])
                    parasite_nodes.append(parasite_emb[i, p])

            # Build sequences with length = number of pairs so attention is non-degenerate
            host_seq = torch.stack(host_nodes, dim=0).unsqueeze(0)      # (1, L, D)
            parasite_seq = torch.stack(parasite_nodes, dim=0).unsqueeze(0)  # (1, L, D)

            cross_attended, _ = self.cross_attention(
                host_seq, parasite_seq, parasite_seq
            )  # (1, L, D)

            # Learnable pooling over the pair tokens (CLS + mapped pairs)
            pool_logits = self.pair_pool_score(cross_attended).squeeze(-1)  # (1, L)
            pool_weights = F.softmax(pool_logits.float(), dim=1).to(dtype=cross_attended.dtype)  # fp32 softmax
            cross_pooled = torch.sum(cross_attended * pool_weights.unsqueeze(-1), dim=1).squeeze(0)  # (D)

            pooled = torch.cat([
                host_seq.mean(dim=1).squeeze(0),
                parasite_seq.mean(dim=1).squeeze(0),
                cross_pooled
            ], dim=-1)  # (3*hidden_dim)

            mapped_pair_features[i] = pooled

        attended_pairs = mapped_pair_features

        # --- sim_time modulation disabled ---
        if sim_time is not None:
            gamma_beta = self.sim_time_fc(sim_time)  # (B, 2 * hidden_dim)
            scale, shift = gamma_beta.chunk(2, dim=-1)  # (B, hidden_dim), (B, hidden_dim)
        
            base = attended_pairs[:, :hidden_dim]
            rest = attended_pairs[:, hidden_dim:]
        
            modulated = base * (1 + scale) + shift
            attended_pairs = torch.cat([modulated, rest], dim=-1)

        out_cospeciation = self.cospeciation_head(attended_pairs)
        out_switch = self.switch_head(attended_pairs)
        outputs = torch.cat([out_cospeciation, out_switch], dim=-1)
        outputs = torch.sigmoid(outputs)
        return outputs
