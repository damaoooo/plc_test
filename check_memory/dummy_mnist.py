import torch
import time
import os
import random
import psutil
import dgl
from torch.utils.data import Dataset, DataLoader


class DummyGraph(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        self.graph_list = self.graph_list
        
    def handle_graph(self, index):
        g:dgl.DGLGraph = self.graph_list[index]
        g = dgl.add_nodes(g, 1024 - g.number_of_nodes())
        g = dgl.add_self_loop(g)
        return g

    def __getitem__(self, index):
        
        graph_list = []
        
        graph_list.append(self.handle_graph(index))
        graph_list.append(self.handle_graph(random.randint(0, 511)))
        
        return graph_list

    def __len__(self):
        return len(self.graph_list)
    
def collate_fn(x):
    batch_size = len(x)
    graph_list = []
    for i in range(batch_size):
        graph_list.append(x[i])
    return {"list": graph_list}
    

def random_graph():
    graph = dgl.graph(([random.randint(0, 999) for x in range(999)] + [999], [random.randint(0, 999) for x in range(999)] + [999]))
    graph.ndata['feat'] = torch.randn(1000, 141)
    graph = dgl.add_nodes(graph, 2)
    graph = dgl.add_self_loop(graph)
    return graph

if __name__ == '__main__':

    graph_list = [random_graph() for x in range(10240)]
    dataset = DummyGraph(graph_list)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=20, collate_fn=collate_fn)
    
    print("Start ...")
    
    a0 = time.time()    
    idx = 0

    for i in data_loader:
        idx += 1
        print(idx)
        print(
            "Memory Usage: %.4f GB"
            % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
        )
    