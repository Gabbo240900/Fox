import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


    
class MSAEncoder(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=8, num_heads=8):
        super(MSAEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=hidden_dim)  # 20 AAs + gap + unknown + virtual node (X)
        self.pool_weights = nn.Linear(hidden_dim, 1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # Reduce from 8 to 4 layers
        self.norm = nn.LayerNorm(hidden_dim)  # Stabilize training
        #self.dropout = nn.Dropout(0.1)
        self.mask_token_id = 22
        

    def forward(self, x):
        mask = torch.rand_like(x.float()) < 0.1  
        x = x.masked_fill(mask, self.mask_token_id)
        x = self.embedding(x)  # (batch, num_leaves+1, seq_len, hidden_dim)

        weights = F.softmax(self.pool_weights(x).squeeze(-1), dim=2)  # (B, N, S)
        x = torch.sum(x * weights.unsqueeze(-1), dim=2)  # (B, N, D)
        x = self.norm(x)
        #x = self.dropout(x)

        cls_token = torch.zeros(x.size(0), 1, x.size(-1), device=x.device)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        # Activation checkpointing across encoder layers to save memory during backprop
        if self.training and hasattr(self.transformer, 'layers') and len(self.transformer.layers) > 1:
            x = checkpoint_sequential(
                self.transformer.layers,
                len(self.transformer.layers),
                x,
                use_reentrant=False
            )
            if getattr(self.transformer, 'norm', None) is not None:
                x = self.transformer.norm(x)
        else:
            x = self.transformer(x)  # (batch, num_leaves+1, hidden_dim)
        return x, x[:, 0]  # Return the full output and CLS token


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

        self.sim_time_fc = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.Identity()
        )
        self.concat_dim = 3 * hidden_dim
        self.cospeciation_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )
        self.switch_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
            host_nodes = []
            parasite_nodes = []
            for h_idx, p_idx in mapping:
                if 0 <= h_idx < host_emb.shape[1] and 0 <= p_idx < parasite_emb.shape[1]:
                    host_nodes.append(host_emb[i, h_idx])
                    parasite_nodes.append(parasite_emb[i, p_idx])
            if host_nodes:
                host_tensor = torch.stack(host_nodes).unsqueeze(1)  # (num_pairs, 1, hidden_dim)
                parasite_tensor = torch.stack(parasite_nodes).unsqueeze(1)  # (num_pairs, 1, hidden_dim)

                attn = self.cross_attention
                cross_attended, _ = attn(host_tensor, parasite_tensor, parasite_tensor)  # (num_pairs, 1, hidden_dim)

                pooled = torch.cat([
                    host_tensor.squeeze(1).mean(dim=0),
                    parasite_tensor.squeeze(1).mean(dim=0),
                    cross_attended.squeeze(1).mean(dim=0)
                ], dim=-1)  # (3*hidden_dim)
                mapped_pair_features[i] = pooled
            else:
                base_host = host_cls[i]
                base_parasite = parasite_cls[i]
                zero_cross = torch.zeros(hidden_dim, device=host_msa.device)
                pooled = torch.cat([base_host, base_parasite, zero_cross], dim=-1)
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
        if self.training:
    	    return outputs  # raw values (unbounded) 
        return outputs.clamp(0.0,1.0)
