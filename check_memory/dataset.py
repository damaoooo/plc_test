import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import random
import lightning.pytorch as pl
import copy
# import pytorch_lightning as pl
# from memory_profiler import profile
from typing import List, Union, Dict, Tuple
import pickle
import json
import os
import gc
import psutil
import time
import numpy as np

import dgl

from multiprocessing import set_start_method

FunctionBody = Dict[str, Union[int, float, str]]
DataIndex = Dict[str, Dict[str, List[FunctionBody]]]

class ASTGraphDataset(Dataset):
    def __init__(
        self, data: list, data_index: DataIndex, max_adj: int, feature_len: int, pool_size: int
    ) -> None:
        super().__init__()
        self.data = data
        self.data_index: DataIndex = data_index
        self.binary_list = list(self.data_index.keys())
        self.phases = [0]
        self.length = self._get_length()
        self.max_adj = max_adj
        self.feature_len = feature_len
        self.pool_size = pool_size


    def __len__(self):
        return self.length

    def _get_length(self):
        length = 0
        phases = [0]
        for key in self.binary_list:
            length += len(self.data_index[key])
            phases.append(phases[-1] + len(self.data_index[key]))
        self.phases = phases
        return length

    def _find_binary_index(self, index):
        for i in range(len(self.phases)):
            if index < self.phases[i]:
                return i - 1, index - self.phases[i - 1]
        return -1, -1

    # @profile
    def __getitem__(self, index):
        # impliment the sliced file reading

        binary_index, function_offset = self._find_binary_index(index)
        binary_name = self.binary_list[binary_index]
        function_name = sorted(list(self.data_index[binary_name].keys()))[function_offset]
        sample_function_list = self.data_index[binary_name][function_name]

        same_pair = random.sample(sample_function_list, 2)
        sample, same_sample = same_pair[0], same_pair[1]

        different_binary_name = random.choice(self.binary_list)
        different_function_name = random.choice(
            list(self.data_index[different_binary_name].keys())
        )

        while (
            different_function_name == function_name
            and different_binary_name == binary_name
        ):
            different_binary_name = random.choice(self.binary_list)
            different_function_name = random.choice(
                list(self.data_index[different_binary_name].keys())
            )

        different_sample = random.choice(
            self.data_index[different_binary_name][different_function_name]
        )

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)

        # Pool candidates
        if self.pool_size:
            pool = self._get_pool(binary_name, function_name)
            pool = [self._to_tensor(x) for x in pool]
            return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample, "label": torch.tensor([0]), "pool": pool}

        return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample, "label": torch.tensor([0])}

    def _get_pool(self, binary_name: str, function_name: str):
        pool = []
        # Get the function pool that does not contain the function_name
        for p in range(self.pool_size):
            pool_binary_name = random.choice(self.binary_list)
            pool_function_name = random.choice(list(self.data_index[pool_binary_name].keys()))
            while (
                pool_binary_name == binary_name and pool_function_name == function_name
            ):
                pool_binary_name = random.choice(self.binary_list)
                pool_function_name = random.choice(
                    list(self.data_index[pool_binary_name].keys())
                )
            pool.append(random.choice(self.data_index[pool_binary_name][pool_function_name]))

        return pool

    # @profile
    def _to_tensor(self, data: dict):
        index = data['index']
        graph: dgl.DGLGraph = self.data[index]
        
        if graph.number_of_nodes() < self.max_adj:
            padding_size = self.max_adj - graph.number_of_nodes()
            graph = dgl.add_nodes(graph, padding_size)
            graph = dgl.add_self_loop(graph)

        return graph


def collate_fn(x):
    batch_size = len(x)
    sample_list = []
    same_sample_list = []
    different_sample_list = []
    for i in range(batch_size):
        sample_list.append(x[i]["sample"])
        same_sample_list.append(x[i]["same_sample"])
        different_sample_list.append(x[i]["different_sample"])
    sample_list = dgl.batch(sample_list)
    same_sample_list = dgl.batch(same_sample_list)
    different_sample_list = dgl.batch(different_sample_list)
    
    if "pool" in x[0]:
        batch_list = []
        for i in range(batch_size):
            pool_list = x[i]["pool"]
            pool_list = dgl.batch(pool_list)
            batch_list.append(pool_list)
        return {"sample": sample_list, "same_sample": same_sample_list, "different_sample": different_sample_list, "label": torch.tensor([0]), "pool": batch_list}
    return {"sample": sample_list, "same_sample": same_sample_list, "different_sample": different_sample_list, "label": torch.tensor([0])}
            



def new_graph():
    graph = dgl.graph(([random.randint(0, 949) for x in range(949)] + [949], [random.randint(0, 949) for x in range(949)] + [949]))
    graph.ndata['feat'] = torch.randn(950, 141)
    graph = dgl.add_nodes(graph, 1)
    graph = dgl.add_self_loop(graph)
    return graph

if __name__ == "__main__":
    # set_start_method('spawn')
    a0 = time.time()

    with open("/home/damaoooo/Downloads/plc_test/dataset/uboot/index_train_data_2.pkl", "rb") as f:
        data_index = pickle.load(f)
        f.close()
        
    feature_len = data_index["feature_len"]
    adj_len = data_index["adj"]
    data_index = data_index["data"]
    
    print(feature_len)

    # data_all = dgl.load_graphs("../dataset/uboot/dgl_graphs.dgl")[0]
    data_all = [new_graph() for x in range(10240)]

    train = ASTGraphDataset(
        data=data_all,
        data_index=data_index,
        max_adj=1000,
        feature_len=141,
        pool_size=50
    )

    train_loader = DataLoader(dataset=train, batch_size=8, shuffle=False, num_workers=20, collate_fn=collate_fn)
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    print(
        "当前进程的内存使用：%.4f GB"
        % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    )
    a2 = a1
    for i in train_loader:
        idx += 1
        print(idx)
        print("Single_time:", time.time() - a2)
        a2 = time.time()
        print(
            "当前进程的内存使用：%.4f GB"
            % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
        )
        # print(i)
        # break
