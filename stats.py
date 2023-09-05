import dgl

import random
import torch
import torch.nn as nn
# import dgl.sparse as dglsp
import dgl.nn.pytorch.conv as conv
import dgl.nn.pytorch.glob as glob




class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, feat_drop: float = 0, attn_drop: float = 0, negative_slope: float = 0.2, residual: bool = False):
        super().__init__()
        self.conv1 = conv.GATv2Conv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual)
        self.conv2 = conv.GATv2Conv(in_feats=out_dim * num_heads, out_feats=out_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual)
        
    def forward(self, g, x):
        x = self.conv1(g, x)
        nodes, heads, feat = x.shape
        x = x.view(nodes, heads * feat)
        x = self.conv2(g, x)
        x = x.squeeze(-2)
        return x
        

class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = torch.nn.ModuleList([MultiHeadGATLayer(500, 500, 6) for x in range(6)])
        self.linear = nn.Linear(500 * 500, 500)
        
    def forward(self, g):
        x = g.ndata['feat']
        for gat in self.gat:
            x = gat(g, x)
        x = x.view(-1, 500 * 500)
        x = self.linear(x)
        return x
    
    
p = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
p.add_nodes(497)
feature = torch.randn(500, 500)
p = dgl.add_self_loop(p)
p.ndata['feat'] = feature

p = p.to('cuda')
model = TestNN()
model = model.cuda()

for i in range(100):
    res = model(p)

    
        