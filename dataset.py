import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import random
import lightning.pytorch as pl
# import pytorch_lightning as pl
from typing import List, Union
import pickle
import json
import os
import gc
import psutil
import time
import numpy as np


class ASTGraphDataset(Dataset):
    def __init__(self, data: dict, max_adj: int, feature_len: int, pool_size: int) -> None:
        super().__init__()
        self.data = data
        self.binary_list = list(data.keys())
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
            length += len(self.data[key])
            phases.append(phases[-1] + len(self.data[key]))
        self.phases = phases
        return length
            
    def _find_binary_index(self, index):
        for i in range(len(self.phases)):
            if index < self.phases[i]:
                return i - 1, index - self.phases[i-1]
        return -1, -1

    # @profile
    def __getitem__(self, index):
        # impliment the sliced file reading
        
        binary_index, function_offset = self._find_binary_index(index)
        binary_name = self.binary_list[binary_index]
        function_name = sorted(list(self.data[binary_name].keys()))[function_offset]
        sample_function_list = self.data[binary_name][function_name]
        
        same_pair = random.sample(sample_function_list, 2)
        sample, same_sample = same_pair[0], same_pair[1]
        
        different_binary_name = random.choice(self.binary_list)
        different_function_name = random.choice(list(self.data[different_binary_name].keys()))
        
        while different_function_name == function_name and different_binary_name == binary_name:
            different_binary_name = random.choice(self.binary_list)
            different_function_name = random.choice(list(self.data[different_binary_name].keys()))
            
        different_sample = random.choice(self.data[different_binary_name][different_function_name])

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)
        
        # Pool candidates
        if self.pool_size:
            pool = self._get_pool(binary_name, function_name)
            pool = [self._to_tensor(x) for x in pool]
            return sample, same_sample, different_sample, torch.tensor([0]), pool

        return sample, same_sample, different_sample, torch.tensor([0])
    
    
    def _get_pool(self, binary_name: str, function_name: str):
        pool = []
        # Get the function pool that does not contain the function_name
        for p in range(self.pool_size):
            pool_binary_name = random.choice(self.binary_list)
            pool_function_name = random.choice(list(self.data[pool_binary_name].keys()))
            while pool_binary_name == binary_name and pool_function_name == function_name:
                pool_binary_name = random.choice(self.binary_list)
                pool_function_name = random.choice(list(self.data[pool_binary_name].keys()))
            pool.append(random.choice(self.data[pool_binary_name][pool_function_name]))
            
        return pool
        

    # @profile
    def _to_tensor(self, data: dict):
        adj = data['adj']
        feature = data['feature']
        feature = torch.FloatTensor(feature)
        # feature = torch.eye(self.feature_len)[feature]
        adj_matrix = torch.zeros([self.max_adj, self.max_adj])
        start = torch.tensor(adj[0])
        end = torch.tensor(adj[1])
        adj_matrix[start, end] = 1

        adj_matrix = (adj_matrix + adj_matrix.T)
        adj_matrix += torch.eye(self.max_adj)

        # For padding
        feature_padder = torch.nn.ZeroPad2d(
            [0, self.feature_len - feature.shape[1], 0, self.max_adj - feature.shape[0]])
        feature = feature_padder(feature)[:self.max_adj][:, :self.feature_len]

        # adj_padder = torch.nn.ZeroPad2d([0, self.max_adj - adj.shape[1], 0, self.max_adj - adj.shape[0]])
        # adj = adj_padder(adj)[: self.max_adj][:, :self.max_adj]

        return feature, adj_matrix


class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "!pairs.pkl", pool_size: int = 0, batch_size: int = 32, num_workers: int = 16, exclude: list = []) -> None:
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

    def _load_dataset_info(self, data_path):
        with open(os.path.join(data_path, "metainfo.json"), 'r') as f:
            content = json.loads(f.read())
            f.close()
        adj_len = content['adj_len']
        feature_len = content['feature_len']
        total_length = content['length']
        offset = content['offset']
        file_list = content['file_list']
        for i in range(len(file_list)):
            file_list[i] = os.path.join(
                os.path.abspath(data_path), file_list[i])
        sorted(file_list)
        return adj_len, feature_len, total_length, offset, file_list

    def _load_pickle_data(self, data_path: str):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            f.close()
        feature_len = data['feature_len']
        adj_len = data['adj']
        data = data['data']
        return adj_len, feature_len, data

    def prepare_data(self):
        adj_len, feature_len, train_data = self._load_pickle_data(
            os.path.join(self.data_path, "all_data.pkl"))

        # Assume train_set and test_set are the same
        _, _, test_data = self._load_pickle_data(
            os.path.join(self.data_path, "test_data.pkl"))

        # total_dataset = ASTGraphDataset(data, max_adj=min(adj_len, 1000), feature_len=feature_len, exclude=self.exclude)

        self.max_length = 1500
        self.feature_length = feature_len

        self.train_set = ASTGraphDataset(
            data=train_data, max_adj=self.max_length, feature_len=self.feature_length, pool_size=self.pool_size)
        self.val_set = ASTGraphDataset(
            data=test_data, max_adj=self.max_length, feature_len=self.feature_length, pool_size=self.pool_size)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)


if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataModule(data_path="coreutil_dataset", pool_size=15, num_workers=16, batch_size=1)
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    a2 = a1
    for i in iter(train):
        idx += 1
        print(idx)
        print("Single_time:", time.time() - a2)
        a2 = time.time()
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
