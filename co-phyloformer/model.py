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
    
class MSAEncoder(nn.Module):
    def __init__(self, hidden_dim=896, num_layers=16, num_heads=8):# increase embedding and layers reduce batch size 
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)
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

        # Mask padding positions (PAD token id = 22) so pooling ignores padded tokens
        pad_mask = (x_ids != 22)  # (B, N, S)
        logits = self.pool_weights(x).squeeze(-1)  # (B, N, S)
        logits = logits.masked_fill(~pad_mask, -1e9)

        weights = F.softmax(logits, dim=2)  # (B, N, S)
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
    def __init__(self, hidden_dim=896, num_layers=16, num_heads=8):
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
        # Single shared head for quick capacity/bottleneck test
        self.cosp_head = nn.Linear(self.concat_dim, 1)
        self.switch_head = nn.Linear(self.concat_dim, 1)
        # Expose last batch mapping stats for debugging/logging
        self.last_pair_counts = None  # Tensor[B]

    def forward(self, host_msa, parasite_msa, mappings, sim_time):
        # Encode host and parasite MSAs

        host_emb, host_cls = self.host_encoder(host_msa)  # (batch, num_leaves+1, hidden_dim), (batch, hidden_dim)
        parasite_emb, parasite_cls = self.parasite_encoder(parasite_msa)  # (batch, num_leaves+1, hidden_dim), (batch, hidden_dim)

        batch_size = host_msa.shape[0]
        hidden_dim = host_emb.shape[-1]

        # Build padded index tensors so we can run cross-attention in batch.
        # Always include CLS at position 0.
        host_indices_list = []
        parasite_indices_list = []
        host_valid_list = []
        parasite_valid_list = []
        pair_counts = []

        for i, mapping in enumerate(mappings):
            h_idx = [0]
            p_idx = [0]
            valid_pairs = 0

            for h_leaf, p_leaf in mapping:
                h = int(h_leaf) + 1
                p = int(p_leaf) + 1
                if 0 <= h < host_emb.shape[1] and 0 <= p < parasite_emb.shape[1]:
                    h_idx.append(h)
                    p_idx.append(p)
                    valid_pairs += 1

            # counts = number of mapped pairs (excluding CLS)
            pair_counts.append(valid_pairs)
            host_indices_list.append(torch.tensor(h_idx, device=host_emb.device, dtype=torch.long))
            parasite_indices_list.append(torch.tensor(p_idx, device=parasite_emb.device, dtype=torch.long))

        max_len = max(t.numel() for t in host_indices_list) if host_indices_list else 1

        host_idx = torch.zeros((batch_size, max_len), device=host_emb.device, dtype=torch.long)
        parasite_idx = torch.zeros((batch_size, max_len), device=parasite_emb.device, dtype=torch.long)
        host_valid = torch.zeros((batch_size, max_len), device=host_emb.device, dtype=torch.bool)
        parasite_valid = torch.zeros((batch_size, max_len), device=host_emb.device, dtype=torch.bool)

        for i in range(batch_size):
            hl = host_indices_list[i].numel()
            pl = parasite_indices_list[i].numel()
            # hl == pl by construction
            host_idx[i, :hl] = host_indices_list[i]
            parasite_idx[i, :pl] = parasite_indices_list[i]
            host_valid[i, :hl] = True
            parasite_valid[i, :pl] = True

        # Gather sequences: (B, L, D)
        b = torch.arange(batch_size, device=host_emb.device)[:, None]
        host_seq = host_emb[b, host_idx]
        parasite_seq = parasite_emb[b, parasite_idx]

        # Cross-attention in batch. Key/value padding mask uses True for pads.
        key_pad = ~parasite_valid
        cross_attended, _ = self.cross_attention(host_seq, parasite_seq, parasite_seq, key_padding_mask=key_pad)

        # Masked means over the sequence dimension
        host_den = host_valid.sum(dim=1, keepdim=True).clamp(min=1)
        parasite_den = parasite_valid.sum(dim=1, keepdim=True).clamp(min=1)

        host_mean = (host_seq * host_valid.unsqueeze(-1)).sum(dim=1) / host_den
        parasite_mean = (parasite_seq * parasite_valid.unsqueeze(-1)).sum(dim=1) / parasite_den
        cross_mean = (cross_attended * host_valid.unsqueeze(-1)).sum(dim=1) / host_den

        mapped_pair_features = torch.cat([host_mean, parasite_mean, cross_mean], dim=-1)  # (B, 3*D)

        attended_pairs = mapped_pair_features

        # Save mapping stats for external logging
        self.last_pair_counts = torch.tensor(pair_counts, device=host_emb.device, dtype=torch.long)

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
