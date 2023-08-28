from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
# import pytorch_lightning as pl

class GraphConvolution(nn.Module):
    """GCN layer"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        Nodes = h.size()[-2]  # N 图的节点数

        a_input = torch.cat([
            # [nodes * nodes, out_features] 其中每个nodes重复nodes次[[a],[a],[a], [b],[b],[b]]
            h.repeat(1, Nodes).view(Nodes * Nodes, -1),
            # [nodes * nodes, out_features], 其中每个nodes重复nodes次[[a], [b], [c], [a], [b], [c]]
            h.repeat(Nodes, 1)
        ], dim=1  # [nodes* nodes, out_features * 2]结果是 [[a, a], [a, b], [a, c] ... ] 类似于全排列
        ).view(Nodes, -1, 2 * self.out_features)
        # [N, N, 2*out_features], 代表每个nodes的全排列单独在一个矩阵[N, 2*out_features]中，有N个
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        # [N, N, out_features * 2] x [out_features * 2, 1] => [N, N, 1]
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e6 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SAGPooling(nn.Module):
    def __init__(self, in_features, dropout):
        super(SAGPooling, self).__init__()
        self.gcn = GraphConvolution(in_features, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        adj = self.normalize_adj(adj)
        x = self.gcn(x, adj)
        x = x.view(1, -1)
        # x = F.softmax(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def normalize_adj(self, adj):
        """compute L=D^-0.5 * (A+I) * D^-0.5"""
        adj += torch.eye(adj.shape[0]).to(device=adj.device)
        degree = adj.sum(1).to(dtype=torch.float)
        d_hat = torch.diag((degree ** (-0.5)).flatten())
        norm_adj = d_hat @ adj @ d_hat
        return norm_adj


class GAT(pl.LightningModule):
    def __init__(self, in_features, hidden_features, out_features, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(in_features, hidden_features,
                                               dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(hidden_features * n_heads, out_features,
                                           dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return x  # log_softmax速度变快，保持数值稳定


class MyModel(pl.LightningModule):
    def __init__(self, in_features, adj_length, hidden_features, out_features, dropout, alpha, n_heads):
        super(MyModel, self).__init__()
        self.gat = GAT(in_features, hidden_features, out_features, dropout, alpha, n_heads)
        self.linear1 = nn.Linear(out_features * adj_length, out_features)
        # self.sagPool = SAGPooling(out_features, dropout)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.414)

    def forward(self, adj, feature):
        x = self.gat(feature, adj)
        x = F.elu(x)
        # x = self.sagPool(x, adj)
        x = self.linear1(x.view(1, -1))

        # x = F.log_softmax(x, dim=-1)
        return x
    

class PLModelForAST(pl.LightningModule):
    def __init__(self, adj_length: int, pool_size: int=0, lr: float=5e-5, in_features=64, hidden_features=128, output_features=64, n_heads=4, dropout=0.6, alpha=0.2, seed=1):
        super().__init__()
        self.lr = lr
        self.pool_size= pool_size
    # def __init__(self, config) -> None:
    #     super().__init__()
    #     self.lr = config['lr']
    #     in_features = config['in_features']
    #     hidden_features = config['hidden_features']
    #     output_features = config['output_features']
    #     n_heads = config['n_heads']
    #     adj_length = config['adj_length']
    #     alpha = config['alpha']
    #     dropout = config['dropout']
        self.seed = seed
        self.my_model = MyModel(in_features=in_features, hidden_features=hidden_features, out_features=output_features, n_heads=n_heads, dropout=dropout, adj_length=adj_length, alpha=alpha)
        self.my_model = torch.compile(self.my_model)
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        if self.pool_size:
            sample, same, diff, label, pool = x
        else:
            sample, same, diff, label = x

        sample_feature, sample_adj = sample
        same_feature, same_adj = same
        diff_feature, diff_adj = diff

        seq = [1, 2, 3]
        random.shuffle(seq)
        for s in seq:
            if s == 1:
                latent_sample = self.my_model(sample_adj.squeeze(0), sample_feature.squeeze(0))
            elif s == 2:
                latent_same = self.my_model(same_adj.squeeze(0), same_feature.squeeze(0))
            else:
                latent_diff = self.my_model(diff_adj.squeeze(0), diff_feature.squeeze(0))

        loss1 = F.cosine_embedding_loss(latent_same, latent_sample, label[0] + 1)
        loss2 = F.cosine_embedding_loss(latent_sample, latent_diff, label[0] - 1)
        
        pool_latents = []
        
        if self.pool_size:
            for k in pool:
                pool_feature, pool_adj = k
                pool_latent = self.my_model(pool_adj.squeeze(0), pool_feature.squeeze(0))
                # use softmax for the pool
                pool_latents.append(pool_latent)
            pool_latents.append(latent_same)
            pool_latents = torch.stack(pool_latents, dim=0)
            similarity = F.cosine_similarity(latent_sample, pool_latents.squeeze(1))
            
            loss3 = F.cross_entropy(similarity, torch.tensor(self.pool_size, dtype=torch.long).to(device=self.device))
            
            return (loss1 + loss2 + loss3, (loss1 + loss2).item(), F.cosine_similarity(latent_same, latent_sample).item() > F.cosine_similarity(latent_diff, latent_sample).item(),
                    F.cosine_similarity(latent_same, latent_sample).item() - F.cosine_similarity(latent_diff, latent_sample).item())

        return (loss1 + loss2, F.cosine_similarity(latent_same, latent_sample).item() > F.cosine_similarity(latent_diff, latent_sample).item(),
                F.cosine_similarity(latent_same, latent_sample).item() - F.cosine_similarity(latent_diff, latent_sample).item())
    
    def get_full_embedding(self, data):
        with torch.no_grad():
            res = []
            name_list = []
            for adj, fea, name in data:
                adj = adj.to(device=self.device)
                fea = fea.to(device=self.device)
                embedding = self.my_model(adj, fea)
                res.append(embedding.squeeze(0).detach().cpu().numpy())
                name_list.append(name)
        res = np.vstack(res)
        return res, name_list


    def training_step(self, batch, batch_idx):
        if self.pool_size:
            loss_all, loss_1v1, ok, _ = self.forward(batch)
            self.log("train_loss_all", loss_all.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.log("train_loss_1v1", loss_1v1, on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.training_step_outputs.append(int(ok))
        else:
            loss_all, ok, _ = self.forward(batch)
            self.log("train_loss_all", loss_all.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.training_step_outputs.append(int(ok))
        return loss_all
    
    def validation_step(self, batch, batch_idx):
        if self.pool_size:
            loss_all, loss_1v1, ok, diff = self.forward(batch)
            self.log("val_loss_all", loss_all.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.validation_step_outputs.append([int(ok), diff, loss_1v1])
        else:
            loss_all, ok, diff = self.forward(batch)
            self.log("val_loss_all", loss_all.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.validation_step_outputs.append([int(ok), diff])
        return loss_all
    
    def on_validation_epoch_end(self):
        acc = np.mean([x[0] for x in self.validation_step_outputs])
        self.log("val_acc", acc.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        
        diff = np.mean([x[1] for x in self.validation_step_outputs])
        self.log("val_diff", diff.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        
        if self.pool_size:
            loss_1v1 = np.mean([x[2] for x in self.validation_step_outputs])
            self.log("val_loss_1v1", loss_1v1.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.validation_step_outputs.clear()
        
    def on_train_epoch_end(self):
        acc = np.mean(self.training_step_outputs)
        self.log("train_acc", acc.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    
