import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import random
import lightning.pytorch as pl

# import pytorch_lightning as pl
from typing import List, Union, Dict, Tuple
import pickle
import json
import os
import gc
import psutil
import time
import numpy as np

import dgl

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
        self.eye = torch.eye(self.max_adj)

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
    return x

class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "!pairs.pkl",
        pool_size: int = 0,
        batch_size: int = 32,
        num_workers: int = 16,
        exclude: list = [],
        k_fold: int = 0,
    ) -> None:
        super().__init__()
        self.data_path = data_path

        self.train_set: Union[Dataset, None] = None
        self.val_set: Union[Dataset, None] = None

        self.pool_size = pool_size

        self.exclude = exclude

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_length = -1
        self.feature_length = -1

        self.k_fold = k_fold

    def _load_dataset_info(self, data_path):
        with open(os.path.join(data_path, "metainfo.json"), "r") as f:
            content = json.loads(f.read())
            f.close()
        adj_len = content["adj_len"]
        feature_len = content["feature_len"]
        total_length = content["length"]
        offset = content["offset"]
        file_list = content["file_list"]
        for i in range(len(file_list)):
            file_list[i] = os.path.join(os.path.abspath(data_path), file_list[i])
        sorted(file_list)
        return adj_len, feature_len, total_length, offset, file_list

    def _load_pickle_data(self, data_path: str):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            f.close()
        feature_len = data["feature_len"]
        adj_len = data["adj"]
        data = data["data"]
        return adj_len, feature_len, data

    def _get_test_index(self, data_dict: dict):
        index = []
        for binary_name in data_dict:
            for function_name in data_dict[binary_name]:
                for function_body in data_dict[binary_name][function_name]:
                    index.append(function_body["index"])

        return index

    def prepare_data(self):
        if self.k_fold:
            train_path = os.path.join(
                self.data_path, f"index_train_data_{self.k_fold}.pkl"
            )
            test_path = os.path.join(
                self.data_path, f"index_test_data_{self.k_fold}.pkl"
            )
        else:
            train_path = os.path.join(self.data_path, "index_train_data.pkl")
            test_path = os.path.join(self.data_path, "index_test_data.pkl")

        adj_len, feature_len, index_train_data = self._load_pickle_data(train_path)

        # Assume train_set and test_set are the same
        _, _, index_test_data = self._load_pickle_data(test_path)

        # total_dataset = ASTGraphDataset(data, max_adj=min(adj_len, 1000), feature_len=feature_len, exclude=self.exclude)

        self.max_length = adj_len
        self.feature_length = feature_len

        # graphs = dgl.load_graphs(os.path.join(self.data_path, "dgl_graphs.dgl"))
        
        all_data, _ = dgl.load_graphs(
            os.path.join(self.data_path, "dgl_graphs.dgl")
        )
        
        self.train_set = ASTGraphDataset(
            data=all_data,
            data_index=index_train_data,
            max_adj=self.max_length,
            feature_len=self.feature_length,
            pool_size=self.pool_size,
        )
        
        self.val_set = ASTGraphDataset(
            data=all_data,
            data_index=index_test_data,
            max_adj=self.max_length,
            feature_len=self.feature_length,
            pool_size=self.pool_size,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn
        )


if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataModule(
        data_path="/home/damaoooo/Downloads/plc_test/dataset/openplc", pool_size=15, num_workers=16, batch_size=1, k_fold=1
    )
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    print(
        "当前进程的内存使用：%.4f GB"
        % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    )
    a2 = a1
    for i in iter(train):
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
