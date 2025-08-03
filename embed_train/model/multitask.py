import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskSAINT(nn.Module):
    def __init__(self, n_numeric_features=50, embed_dim=64, num_layers=2, num_heads=4, ff_dim=256, dropout=0.2):
        super().__init__()
        self.backbone = __import__('embed_train.model.architecture', fromlist=['SAINTBackbone']).SAINTBackbone(
            n_numeric_features, embed_dim, num_layers, num_heads, ff_dim, dropout
        )
        self.n_cols = n_numeric_features
        self.embed_dim = embed_dim
        self.mask_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        self.next_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_numeric_features)
        )
    def forward_backbone(self, numeric_x):
        return self.backbone(numeric_x)
    def forward_masked(self, numeric_x):
        out = self.forward_backbone(numeric_x)
        pred = self.mask_fc(out).squeeze(-1)
        return pred
    def forward_nextpredict(self, numericA):
        outA = self.forward_backbone(numericA)
        embA = outA.mean(dim=1)
        predB = self.next_fc(embA)
        return predB
    def get_embedding(self, numeric_x):
        out = self.forward_backbone(numeric_x)   # => (B,C,d)
        emb = out.mean(dim=1)                   # => (B,d)
        return emb
    def row_embedding(self, numeric_rows):
        # Handle batch inputs correctly
        emb = self.get_embedding(numeric_rows)   # => (batch_size, d)
        return emb

def contrastive_triplet_loss(anchor, positive, negative, margin=0.1):
    pos_score = F.cosine_similarity(anchor, positive, dim=1)
    neg_score = F.cosine_similarity(anchor, negative, dim=1)
    losses = F.relu(neg_score - pos_score + margin)
    return losses.mean()


def add_gaussian_noise(tensor, std=0.01):
    if std <= 0.0:
        return tensor
    noise = torch.randn_like(tensor) * std
    return tensor + noise