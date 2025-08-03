import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
    def forward(self, x, padmask=None):
        return self.layer(x, src_key_padding_mask=padmask)

class SAINTBackbone(nn.Module):
    def __init__(self, n_numeric_features, embed_dim=128, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.n_cols = n_numeric_features
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.col_embed = nn.Parameter(torch.randn(n_numeric_features, embed_dim))
        self.row_blocks = nn.ModuleList([MultiHeadAttentionBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.col_blocks = nn.ModuleList([MultiHeadAttentionBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, C = x.shape
        ce = self.col_embed.unsqueeze(0).expand(B, C, self.embed_dim)
        out = x.unsqueeze(-1) * ce
        out = self.ln(out)
        out = self.dropout(out)
        for i in range(self.num_layers):
            out = self.row_blocks[i](out)
            out_t = out.transpose(0, 1)
            out_t = self.col_blocks[i](out_t)
            out = out_t.transpose(0, 1)
        out = self.ln(out)
        return out