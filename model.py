

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv
import numpy as np


def similarity_score(x, y):
    distance = torch.norm(x - y, dim=-1)
    score = 1 / (1 + distance)
    return score


def pearson_score(a, b):
    a_mu = (a - a.mean(dim=-1).unsqueeze(-1))
    b_mu = (b - b.mean(dim=-1).unsqueeze(-1))
    return (a_mu * b_mu).clamp(-0x7FFF, 0x7FFF).mean(dim=-1) / (
            a.std(correction=0, dim=-1) * b.std(correction=0, dim=-1))


class BaseModel(nn.Module):
    def __init__(self, in_feature: int, hidden_feature: int, out_feature: int, num_heads: int, dropout: float,
                 alpha: float, adj_len: int):
        super().__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.adj_len = adj_len

        self.conv1 = GATv2Conv(in_feats=self.in_feature, out_feats=self.hidden_feature, num_heads=self.num_heads,
                               feat_drop=self.dropout, attn_drop=self.dropout, negative_slope=self.alpha)
        self.conv2 = GATv2Conv(in_feats=self.hidden_feature * self.num_heads, out_feats=self.hidden_feature,
                               num_heads=1,
                               feat_drop=self.dropout, attn_drop=self.dropout, negative_slope=self.alpha)
        self.nlp = nn.Linear(self.hidden_feature * self.adj_len, 128)
        # self.read_out = Set2Set(self.hidden_feature, n_iters=3, n_layers=3)

    def forward(self, g):
        h = g.ndata['feat']
        batch_size = h.shape[0] // self.adj_len
        h = self.conv1(g, h)
        nodes, heads, output_features = h.shape
        h = torch.reshape(h, (nodes, heads * output_features))
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.squeeze(-2)
        h = F.elu(h)
        h = self.nlp(h.view(batch_size, -1))
        return h


class ReGraphModel(nn.Module):
    def __init__(self, base_model: BaseModel, pool_size: int = 0):
        super().__init__()
        self.base_mode = base_model
        self.pool_size = pool_size

    def forward(self, x):
        # FIXME: maybe remove this if not needed without lightning
        torch.cuda.empty_cache()
        sample, same, diff, label, pool = x['sample'], x['same_sample'], x['different_sample'], x['label'], x['pool']

        sample_vector: torch.Tensor = self.base_mode(sample)
        same_vector: torch.Tensor = self.base_mode(same)
        diff_vector: torch.Tensor = self.base_mode(diff)

        # Pearson(sample, same) should be close to 1
        # Pearson(sample, diff) should be close to 0
        loss_basic = (1 - abs(pearson_score(sample_vector, same_vector))) + abs(pearson_score(sample_vector, diff_vector))

        batch_size, output_size = same_vector.shape[0], same_vector.shape[1]

        pool_vectors = [self.base_mode(pool[b]) for b in range(batch_size)]
        pool_vectors = torch.vstack(pool_vectors)
        pool_vectors = pool_vectors.view(batch_size, self.pool_size, output_size)
        pool_vectors = torch.concat([pool_vectors, same_vector.unsqueeze(1)], dim=1)
        pool_similarity = abs(pearson_score(sample_vector.unsqueeze(1), pool_vectors))

        loss_pool = F.cross_entropy(pool_similarity, torch.tensor([self.pool_size] * batch_size, dtype=torch.long).to(device=same_vector.device))

        if not self.training:
            # Calculate diff and acc
            with torch.no_grad():
                diff_score: torch.Tensor = pearson_score(sample_vector, diff_vector).detach().cpu().numpy()
                same_score: torch.Tensor = pearson_score(sample_vector, same_vector).detach().cpu().numpy()
                diff_value: np.array = (same_score - diff_score).mean()
            return loss_basic, loss_pool, diff_value
        else:
            return loss_basic, loss_pool
