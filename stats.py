import dgl

import random
import torch
import dgl.sparse as dglsp
import dgl.nn as dglnn


p = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
feature = [[random.randint(0, 1) for x in range(100)] for y in range(3)]
p.ndata['feat'] = torch.tensor(feature).to_sparse()