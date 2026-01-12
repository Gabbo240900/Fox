import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AxialMSAEncoderLayer(nn.Module):
    """Axial attention over an MSA tensor.

    x is (B, N, S, D)
      - residue attention: attention over S within each sequence (per N)
      - leaf attention: attention over N at each residue position (per S)

    Uses PyTorch scaled_dot_product_attention (FlashAttention when available).
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Residue-axis attention (over S)
        self.norm_residue = nn.LayerNorm(hidden_dim)
        self.qkv_residue = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj_residue = nn.Linear(hidden_dim, hidden_dim)

        # Leaf-axis attention (over N)
        self.norm_leaf = nn.LayerNorm(hidden_dim)
        self.qkv_leaf = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj_leaf = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _sdp_attn(self, q, k, v):
        # q,k,v: (B, H, L, Hd)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, S, D)
        B, N, S, D = x.shape

        # --- Residue attention (over S) ---
        xr = self.norm_residue(x)
        xr = xr.contiguous().view(B * N, S, D)  # (B*N, S, D)

        qkv = self.qkv_residue(xr)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B * N, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, H, S, Hd)
        k = k.view(B * N, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * N, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._sdp_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B * N, S, D)
        out = self.proj_residue(out)
        out = out.view(B, N, S, D)
        x = x + out

        # --- Leaf attention (over N) ---
        xl = self.norm_leaf(x)
        xl = xl.permute(0, 2, 1, 3).contiguous().view(B * S, N, D)  # (B*S, N, D)

        qkv = self.qkv_leaf(xl)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B * S, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B*S, H, N, Hd)
        k = k.view(B * S, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * S, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._sdp_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B * S, N, D)
        out = self.proj_leaf(out)
        out = out.view(B, S, N, D).permute(0, 2, 1, 3).contiguous()  # back to (B, N, S, D)
        x = x + out

        # --- FFN ---
        x = x + self.ff(self.norm_ff(x))
        return x
    
class MSAEncoder(nn.Module):
    def __init__(self, hidden_dim=640, num_layers=12, num_heads=8):# increase embedding and layers reduce batch size 
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool_weights = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            AxialMSAEncoderLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        #self.dropout = nn.Dropout(0.1)
        self.mask_token_id = 22
        

    def forward(self, x):
        # x: (B, N, S) token ids
        x_ids = x
        x = self.embedding(x_ids)  # (B, N, S, D)

        # Axial attention over (S then N) while still 4D
        for layer in self.layers:
            x = layer(x)  # (B, N, S, D)

        # Pool over sequence length S (ignore PAD token id = 22)
        pad_mask = (x_ids != 22)  # (B, N, S)
        logits = self.pool_weights(x).squeeze(-1)  # (B, N, S)
        logits = logits.masked_fill(~pad_mask, -1e9)
        weights = F.softmax(logits, dim=2)  # (B, N, S)
        x = torch.sum(x * weights.unsqueeze(-1), dim=2)  # (B, N, D)

        x = self.norm(x)

        # Add CLS token after pooling so downstream code stays unchanged
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        x = self.final_norm(x)
        return x, x[:, 0]

class Cophyloformer(nn.Module):
    def __init__(self, hidden_dim=640, num_layers=12, num_heads=8):
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

            pooled = torch.cat([
                host_seq.mean(dim=1).squeeze(0),
                parasite_seq.mean(dim=1).squeeze(0),
                cross_attended.mean(dim=1).squeeze(0)
            ], dim=-1)  # (3*hidden_dim)

            mapped_pair_features[i] = pooled

        attended_pairs = mapped_pair_features

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
