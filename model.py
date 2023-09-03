from typing import Any, Optional

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
# import pytorch_lightning as pl

import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn.pytorch.glob import Set2Set


class MyModel(nn.Module):
    def __init__(self, in_feature: int, hidden_feature: int, out_feature: int, num_heads: int, dropout: float, alpha: float):
        super().__init__()
        
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        self.conv1 = GATv2Conv(in_feats=self.in_feature, out_feats=self.hidden_feature, num_heads=self.num_heads, 
                               feat_drop=self.dropout, attn_drop=self.dropout, negative_slope=self.alpha)
        self.conv2 = GATv2Conv(in_feats=self.hidden_feature * self.num_heads, out_feats=self.hidden_feature, num_heads=1, 
                               feat_drop=self.dropout, attn_drop=self.dropout, negative_slope=self.alpha)
        # self.nlp = nn.Linear(64*1000, 128)
        self.read_out = Set2Set(self.hidden_feature, n_iters=3, n_layers=3)
        
    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h)
        nodes, heads, output_features = h.shape
        h = torch.reshape(h, (nodes, heads * output_features))
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.squeeze(-2)
        h = F.elu(h)
        h = self.read_out(g, h)
        return h

    

class PLModelForAST(pl.LightningModule):
    def __init__(self, adj_length: int, pool_size: int=0, lr: float=5e-5, in_features=64, hidden_features=128, output_features=64, n_heads=4, dropout=0.6, alpha=0.2, seed=1, data_path=''):
        super().__init__()
        self.lr = lr
        self.pool_size= pool_size
        self.adj_length = adj_length
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
        self.my_model = MyModel(in_feature=in_features, hidden_feature=hidden_features, out_feature=output_features, num_heads=n_heads, dropout=dropout, alpha=alpha)
        # self.my_model = torch.compile(self.my_model)
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.data_path = data_path
        self.save_hyperparameters()

    def forward(self, x):
        x = x[0]
        if self.pool_size:
            sample, same, diff, label, pool = x['sample'], x['same_sample'], x['different_sample'], x['label'], x['pool']
        else:
            sample, same, diff, label = x['sample'], x['same_sample'], x['different_sample'], x['label']

        seq = [1, 2, 3]
        random.shuffle(seq)
        for s in seq:
            if s == 1:
                latent_sample = self.my_model(sample)
            elif s == 2:
                latent_same = self.my_model(same)
            else:
                latent_diff = self.my_model(diff)

        loss1 = F.cosine_embedding_loss(latent_same, latent_sample, label + 1)
        loss2 = F.cosine_embedding_loss(latent_sample, latent_diff, label - 1)
        
        pool_latents = []
        
        if self.pool_size:
            for k in pool:
                pool_latent = self.my_model(k)
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

    
