
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



class AxialMSABlockLite(nn.Module):

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

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: (B, N, D)
        key_padding_mask: (B, N) with True where token is PAD/invalid (should be ignored)
        """
        B, N, D = x.shape
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Hd)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)


        # SDPA has produced NaNs in bf16 on this environment when masking is used.
        # Use an explicit fp32 attention implementation for stability.
        q_fp32 = q.float()  # (B, H, N, Hd)
        k_fp32 = k.float()  # (B, H, N, Hd)
        v_fp32 = v.float()  # (B, H, N, Hd)

        # scores: (B, H, N, N)
        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)

        if key_padding_mask is not None:
            # key_padding_mask: (B, N) True = ignore key
            km = key_padding_mask[:, None, None, :]  # (B,1,1,N)
            # if a row has all keys masked, unmask CLS key (pos 0)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                km = km.clone()
                km[all_masked, :, :, 0] = False
            scores = scores.masked_fill(km, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_fp32)  # (B, H, N, Hd)

        out = out.to(dtype=q.dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, D)

        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x
    
class MSAEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=8, num_heads=8, axial_layers=1, leaf_attn_max_leaves=128):# increase embedding and layers reduce batch size 
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)
        # MSA-Transformer-style axial attention blocks BEFORE pooling
        self.axial_layers = int(axial_layers)
        self.axial_blocks = nn.ModuleList([
            AxialMSABlockLite(hidden_dim, num_heads, ff_mult=4, dropout=0.0, leaf_attn_max_leaves=leaf_attn_max_leaves)
            for _ in range(self.axial_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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

        # leaf_present: (B, N) True if leaf has any non-PAD residue
        leaf_present = (x_ids != 22).any(dim=2)  # (B, N)

        # key_padding_mask over sequence (CLS + leaves): True means PAD/ignore
        cls_present = torch.ones((leaf_present.size(0), 1), device=leaf_present.device, dtype=torch.bool)
        seq_present = torch.cat([cls_present, leaf_present], dim=1)      # (B, N+1)
        key_padding_mask = ~seq_present                                  # (B, N+1)

        for li, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x, x[:, 0]

class Cophyloformer(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=8, num_heads=8):
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

        
        self.sim_time_fc = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.Identity()
        )
        self.concat_dim = 4 * hidden_dim
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
        
        # ---- Global channel: CLS ↔ CLS cross-attention (batched, 1 token) ----
        global_cross, _ = self.cross_attention(
            host_cls.unsqueeze(1),         # (B,1,D) queries
            parasite_cls.unsqueeze(1),     # (B,1,D) keys
            parasite_cls.unsqueeze(1)      # (B,1,D) values
        )
        global_cross = global_cross.squeeze(1)  # (B,D)

        batch_size = host_msa.shape[0]
        hidden_dim = host_emb.shape[-1]
        max_pairs = max((len(m) for m in mappings), default=0)
        L = max_pairs + 1  # +1 for CLS

        device = host_emb.device

        # Store leaf indices for gather (offset by +1 for CLS). Use -1 for PAD.
        host_idx = torch.full((batch_size, max_pairs), -1, device=device, dtype=torch.long)
        para_idx = torch.full((batch_size, max_pairs), -1, device=device, dtype=torch.long)
        for i, mapping in enumerate(mappings):
            if len(mapping) == 0:
                continue
            # mapping entries are (h_leaf_idx, p_leaf_idx) with 0-based leaf indices
            h = torch.as_tensor([hp[0] for hp in mapping], device=device, dtype=torch.long) + 1
            p = torch.as_tensor([hp[1] for hp in mapping], device=device, dtype=torch.long) + 1
            # clip in case any mapping index is out of bounds
            h = h.clamp(min=0, max=host_emb.shape[1] - 1)
            p = p.clamp(min=0, max=parasite_emb.shape[1] - 1)

            host_idx[i, : h.numel()] = h
            para_idx[i, : p.numel()] = p

        # Gather mapped leaf embeddings in batch
        # (B, max_pairs) -> (B, max_pairs, D)
        safe_host_idx = host_idx.clamp(min=0)
        safe_para_idx = para_idx.clamp(min=0)

        host_gather = host_emb.gather(1, safe_host_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))
        para_gather = parasite_emb.gather(1, safe_para_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))

        # Zero-out padded gathered slots
        host_gather = host_gather.masked_fill((host_idx == -1).unsqueeze(-1), 0)
        para_gather = para_gather.masked_fill((para_idx == -1).unsqueeze(-1), 0)

        # Build sequences: [CLS] + mapped leaves
        host_seq = torch.cat([host_emb[:, 0:1, :], host_gather], dim=1)          # (B, L, D)
        parasite_seq = torch.cat([parasite_emb[:, 0:1, :], para_gather], dim=1)  # (B, L, D)

        # Mask padded pair positions (CLS is always present)
        pair_present = torch.ones((batch_size, L), device=device, dtype=torch.bool)
        if max_pairs > 0:
            pair_present[:, 1:] = (host_idx != -1) & (para_idx != -1)

        # Cross-attention in batch
        # key_padding_mask: True = ignore (mask out)
        kv_pad_mask = ~pair_present
        cross_attended, _ = self.cross_attention(
            host_seq, parasite_seq, parasite_seq, key_padding_mask=kv_pad_mask
        )  # (B, L, D)

        # Learnable pooling over pair tokens (mask padded tokens)
        pool_logits = self.pair_pool_score(cross_attended).squeeze(-1)  # (B, L)
        pool_logits = pool_logits.masked_fill(~pair_present, -1e4)
        pool_weights = F.softmax(pool_logits.float(), dim=1).to(dtype=cross_attended.dtype)
        pool_weights = pool_weights * pair_present.to(dtype=cross_attended.dtype)
        denom = pool_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pool_weights = pool_weights / denom
        cross_pooled = torch.sum(cross_attended * pool_weights.unsqueeze(-1), dim=1)  # (B, D)


        attended_pairs = torch.cat([host_cls, parasite_cls, global_cross, cross_pooled], dim=-1)  # (B, 4*hidden_dim)

        if sim_time is not None:
            gamma_beta = self.sim_time_fc(sim_time)  # (B, 2 * hidden_dim)
            scale, shift = gamma_beta.chunk(2, dim=-1)  # (B, hidden_dim), (B, hidden_dim)

            base = attended_pairs[:, :hidden_dim]          # host_cls
            rest = attended_pairs[:, hidden_dim:]          # parasite_cls + global + local

            modulated = base * (1 + scale) + shift
            attended_pairs = torch.cat([modulated, rest], dim=-1)

        out_cospeciation = self.cospeciation_head(attended_pairs)
        out_switch = self.switch_head(attended_pairs)
        outputs = torch.cat([out_cospeciation, out_switch], dim=-1)
        outputs = torch.sigmoid(outputs)
        return outputs
