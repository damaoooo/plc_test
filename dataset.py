import copy
import json
import os
import pickle
import random
import time
from typing import List, Union, Dict

import dgl
import psutil

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

FunctionBody = Dict[str, Union[int, float, str]]
BinaryData = Dict[str, List[FunctionBody]]
DataIndex = Dict[str, BinaryData]


class ASTGraphDataset(Dataset):
    def __init__(
            self, data: list, data_index: DataIndex, max_adj: int, feature_len: int, pool_size: int) -> None:
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
        # implement the sliced file reading

        binary_index, function_offset = self._find_binary_index(index)
        assert binary_index != -1 and function_offset != -1

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

        different_sample = self._to_tensor(different_sample)
        sample_dict = sample
        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)

        pool = self._get_pool(sample=sample_dict)
        pool = [self._to_tensor(x) for x in pool]
        return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample,
                "label": torch.tensor([0]), "pool": pool}

    def _get_pool(self, sample: dict):
        pool = []
        # Get the function pool that does not contain the function_name
        for p in range(self.pool_size):
            pool_binary_name = random.choice(self.binary_list)
            pool_function_name = random.choice(list(self.data_index[pool_binary_name].keys()))

            while (
                    pool_binary_name == sample['binary'] and pool_function_name == sample['name']
            ):
                pool_binary_name = random.choice(self.binary_list)
                pool_function_name = random.choice(
                    list(self.data_index[pool_binary_name].keys())
                )

            pool_item = random.choice(self.data_index[pool_binary_name][pool_function_name])
            pool.append(pool_item)

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

    batch_list = []
    for i in range(batch_size):
        pool_list = x[i]["pool"]
        pool_list = dgl.batch(pool_list)
        batch_list.append(pool_list)
    return {"sample": sample_list, "same_sample": same_sample_list, "different_sample": different_sample_list,
            "label": torch.tensor([0]), "pool": batch_list}


def _load_pickle_data(data_path: str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        f.close()
    feature_len = data["feature_len"]
    adj_len = data["adj"]
    data = data["data"]
    return adj_len, feature_len, data


class ASTGraphDataLoader:
    def __init__(self, data_path: str, pool_size: int, k_fold: int = 0, batch_size: int = 4, num_workers: int = 4,
                 pin_memory: bool = True, prefetch_factor: int = 2):

        self.data_path = data_path
        self.pool_size = pool_size
        self.k_fold = k_fold

        self.train_set, self.val_set = self._load_data_from_path()
        self.train_loader, self.val_loader = self.get_loader(batch_size=batch_size, num_workers=num_workers,
                                                             pin_memory=pin_memory, prefetch_factor=prefetch_factor)

        self.adj_len = self.train_set.max_adj
        self.feature_len = self.train_set.feature_len

    def _load_data_from_path(self):
        if self.k_fold:
            train_path = os.path.join(self.data_path, f"index_train_data_{self.k_fold}.pkl")
            test_path = os.path.join(self.data_path, f"index_test_data_{self.k_fold}.pkl")
        else:
            train_path = os.path.join(self.data_path, "index_train_data.pkl")
            test_path = os.path.join(self.data_path, "index_test_data.pkl")

        adj_len, feature_len, train_data = _load_pickle_data(train_path)
        _, _, test_data = _load_pickle_data(test_path)

        all_data, _ = dgl.load_graphs(os.path.join(self.data_path, "dgl_graphs.dgl"))
        train_set = ASTGraphDataset(
            data=all_data,
            data_index=train_data,
            max_adj=adj_len,
            feature_len=feature_len,
            pool_size=self.pool_size,
        )

        val_set = ASTGraphDataset(
            data=all_data,
            data_index=test_data,
            max_adj=adj_len,
            feature_len=feature_len,
            pool_size=self.pool_size,
        )

        return train_set, val_set

    def get_loader(self, batch_size: int, num_workers: int, pin_memory: bool = True, prefetch_factor: int = 2):
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        val_loader = DataLoader(
            self.val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        return train_loader, val_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader


if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataLoader(data_path="dataset/uboot_uncompress", pool_size=50, batch_size=4, num_workers=8, k_fold=5, )
    # p = ASTGraphDataModule(data_path="dataset/uboot_dataset", pool_size=50, batch_size=10, num_workers=4, k_fold=1)
    train = p.get_train_loader()
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    print(
        "当前进程的内存使用：%.4f GB"
        % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    )
    a2 = a1
    for i in train:
        idx += 1
        print(idx)
        print("Single_time:", time.time() - a2)
        a2 = time.time()
        # print(i)
        # break
    a3 = time.time()
    print("总共用时：", a3 - a0)
