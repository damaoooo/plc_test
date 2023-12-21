import copy
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
import redis

import dgl

FunctionBody = Dict[str, Union[int, float, str]]
DataIndex = Dict[str, Dict[str, List[FunctionBody]]]

class ASTGraphDataset(Dataset):
    def __init__(
        self, data: list, data_index: DataIndex, max_adj: int, feature_len: int, pool_size: int, environment: tuple = None
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
        self.environment = environment

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
            if self.environment:
                functions = self.data_index[pool_binary_name][pool_function_name]
                for func in functions:
                    if func["arch"] == self.environment[0] and func["opt"] == self.environment[1]:
                        pool.append(func)
                        break
                
            else:
                pool.append(random.choice(self.data_index[pool_binary_name][pool_function_name]))

        return pool
    
    def _get_env_pool(self, binary_name: str, function_name: str):
        pool = []
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
            

class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "!pairs.pkl",
        pool_size: int = 0,
        batch_size: int = 32,
        num_workers: int = 16,
        exclude: list = [],
        k_fold: int = 0,
        exclusive_arch: str = None,
        exclusive_opt: str = None,
        redis: bool = False,
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
        
        self.redis = redis
        
        if exclusive_arch and exclusive_opt:
            self.environment = (exclusive_arch, exclusive_opt)
        else:
            self.environment = None

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
    
    def filter_environment_for_train(self, data_dict: dict, environment: tuple):
        # Need to modify to support the binary mode
        target_arch, target_opt = environment
        filtered_dict = copy.deepcopy(data_dict)
        for binary_name in data_dict:
            for function_name in data_dict[binary_name]:
                filtered_dict[binary_name][function_name] = [x for x in filtered_dict[binary_name][function_name] if x["arch"] != target_arch or x["opt"] != target_opt]
        
        bad_function_name_list = []
        for binary_name in filtered_dict:
            for function_name in filtered_dict[binary_name]:
                if len(filtered_dict[binary_name][function_name]) < 2:
                    bad_function_name_list.append(function_name)
        for bad_function_name in bad_function_name_list:
            del filtered_dict[binary_name][bad_function_name]
            
        bad_binary_name_list = []
        for binary_name in filtered_dict:
            if len(filtered_dict[binary_name]) < 2:
                bad_binary_name_list.append(binary_name)
        for bad_binary_name in bad_binary_name_list:
            del filtered_dict[bad_binary_name]
            
        return filtered_dict
    
    def filter_environment_for_test(self, data_dict: dict, environment: tuple):
        target_arch, target_opt = environment
        # Pick out the functions that don't contain the target environment
        # Need to modify to support the binary mode
        kick_function_name_list = []
        for binary_name in data_dict:
            for function_name in data_dict[binary_name]:
                is_kick = True
                for function_body in data_dict[binary_name][function_name]:
                    if function_body["arch"] == target_arch and function_body["opt"] == target_opt:
                        is_kick = False
                if is_kick:
                    kick_function_name_list.append(function_name)
                    
        for kick_function_name in kick_function_name_list:
            del data_dict[binary_name][kick_function_name]
            
        kick_binary_name_list = []
        for binary_name in data_dict:
            if len(data_dict[binary_name]) < 2:
                kick_binary_name_list.append(binary_name)
        for kick_binary_name in kick_binary_name_list:
            del data_dict[kick_binary_name]

        for binary_name in data_dict:
            assert len(data_dict[binary_name]) >= self.pool_size + 1, f"Binary {binary_name} has less than {self.pool_size + 1} functions"
        return data_dict
        

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
        
        if self.environment:
            index_train_data = self.filter_environment_for_train(index_train_data, self.environment)
            index_test_data = self.filter_environment_for_test(index_test_data, self.environment)

        self.max_length = adj_len
        self.feature_length = feature_len
        
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
            environment=self.environment
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            # prefetch_factor=16
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            # prefetch_factor=16
        )


class ASTGraphRedisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str = "uboot",
        pool_size: int = 0,
        batch_size: int = 32,
        num_workers: int = 16,
        k_fold: int = 0,
        data_path: str = "dataset/uboot_dataset",
        
    ):
        super().__init__()
        self.data_name = data_name
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.redis = redis.ConnectionPool(host="localhost", port=6379, db=0)
        self.k_fold = k_fold
        self.data_path = data_path
        
        
    def prepare_data(self):
        pass
        

if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataModule(
        data_path="dataset/uboot_dataset", pool_size=15, num_workers=16, batch_size=1, k_fold=1, environment=("arm", "mix")
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
