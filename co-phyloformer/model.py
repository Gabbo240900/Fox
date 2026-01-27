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
    def __init__(self, hidden_dim=512, num_layers=4, num_heads=8, residue_layers=1, residue_ff_mult=4, residue_dropout=0.0):# increase embedding and layers reduce batch size 
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)

        # Residue-level contextualization (per leaf, over S) before pooling.
        # This adds MSA-Transformer-like power without positional encodings.
        self.residue_layers = residue_layers
        if residue_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * residue_ff_mult,
                dropout=residue_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.residue_encoder = nn.TransformerEncoder(enc_layer, num_layers=residue_layers)
        else:
            self.residue_encoder = None

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

        # Residue-level Transformer over S for each leaf independently.
        # We reshape (B, N, S, D) -> (B*N, S, D) and use padding mask so PAD tokens are ignored.
        if self.residue_encoder is not None:
            B, N, S, D = x.shape
            x_bn = x.view(B * N, S, D)
            pad_mask_bn = (x_ids.view(B * N, S) == 22)  # True where PAD
            x_bn = self.residue_encoder(x_bn, src_key_padding_mask=pad_mask_bn)
            x = x_bn.view(B, N, S, D)

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
    def __init__(self, hidden_dim=512, num_layers=4, num_heads=8):
        super(Cophyloformer, self).__init__()
        # Store hyperparameters for W&B logging
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        #self.dropout = 0.1
        self.embedding_dim = hidden_dim
        
        # Add residue-level context before pooling (set residue_layers=0 to disable)
        self.host_encoder = MSAEncoder(hidden_dim, num_layers, num_heads, residue_layers=2)
        self.parasite_encoder = MSAEncoder(hidden_dim, num_layers, num_heads, residue_layers=2)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Bi-directional cross-attention refinement over mapped pair tokens
        self.pair_fuse = nn.Linear(2 * hidden_dim, hidden_dim)

        pair_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # 2 lightweight layers over the (CLS + mapped pairs) token sequence
        self.pair_encoder = nn.TransformerEncoder(pair_enc_layer, num_layers=1)

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

        # --- Batched cross-attention over variable-length (host, parasite) mapped pairs ---
        # We pad to L_max within the batch and use key_padding_mask so attention ignores padding.
        # This removes the per-sample Python loop around MultiheadAttention (major speedup on multi-GPU).

        # Compute per-sample lengths: always include CLS↔CLS plus valid mapping pairs
        lengths = []
        valid_pairs_per_sample = []
        for i, mapping in enumerate(mappings):
            pairs = []
            for h_idx, p_idx in mapping:
                h = h_idx +1 
                p = p_idx +1 
                if 0 <= h < host_emb.shape[1] and 0 <= p < parasite_emb.shape[1]:
                    pairs.append((h, p))
            valid_pairs_per_sample.append(pairs)
            lengths.append(1 + len(pairs))  # +1 for CLS↔CLS

        L_max = max(lengths) if lengths else 1

        host_seq_batch = torch.zeros(
            batch_size, L_max, hidden_dim,
            device=host_emb.device,
            dtype=host_emb.dtype,
        )
        parasite_seq_batch = torch.zeros(
            batch_size, L_max, hidden_dim,
            device=parasite_emb.device,
            dtype=parasite_emb.dtype,
        )

        # valid_mask: True where token is real, False where it is padding
        valid_mask = torch.zeros(
            batch_size, L_max,
            device=host_emb.device,
            dtype=torch.bool,
        )

        # Fill CLS↔CLS at position 0 for all samples
        host_seq_batch[:, 0] = host_emb[:, 0]
        parasite_seq_batch[:, 0] = parasite_emb[:, 0]
        valid_mask[:, 0] = True

        # Fill mapped leaf↔leaf pairs
        for i, pairs in enumerate(valid_pairs_per_sample):
            pos = 1
            for h, p in pairs:
                if pos >= L_max:
                    break
                host_seq_batch[i, pos] = host_emb[i, h]
                parasite_seq_batch[i, pos] = parasite_emb[i, p]
                valid_mask[i, pos] = True
                pos += 1

        # key_padding_mask expects True for positions that should be ignored
        key_padding_mask = ~valid_mask  # (B, L_max)

        # Bi-directional cross-attention: host→parasite and parasite→host
        host_to_para, _ = self.cross_attention(
            host_seq_batch,
            parasite_seq_batch,
            parasite_seq_batch,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, L_max, D)

        para_to_host, _ = self.cross_attention(
            parasite_seq_batch,
            host_seq_batch,
            host_seq_batch,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, L_max, D)

        # Fuse both directions into a single pair-token representation
        pair_tokens = self.pair_fuse(torch.cat([host_to_para, para_to_host], dim=-1))  # (B, L_max, D)

        # Refine pair tokens with a small Transformer over the token axis (CLS + pairs)
        # TransformerEncoder uses src_key_padding_mask=True to ignore pads.
        pair_tokens = self.pair_encoder(pair_tokens, src_key_padding_mask=key_padding_mask)  # (B, L_max, D)

        def masked_mean(x, mask):
            # x: (B, L, D), mask: (B, L) with True for valid
            mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
            summed = (x * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            return summed / denom

        host_pooled = masked_mean(host_seq_batch, valid_mask)          # (B, D)
        parasite_pooled = masked_mean(parasite_seq_batch, valid_mask)  # (B, D)
        cross_pooled = masked_mean(pair_tokens, valid_mask)            # (B, D)

        attended_pairs = torch.cat([host_pooled, parasite_pooled, cross_pooled], dim=-1)  # (B, 3*D)

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
