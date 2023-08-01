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
        self.function_list = list(data.keys())
        self.max_adj = max_adj
        self.feature_len = feature_len
        self.pool_size = pool_size

    def __len__(self):
        return len(self.data)

    # @profile
    def __getitem__(self, index):
        # impliment the sliced file reading
        sample_function_list = self.function_list[index]
        same_pair = random.sample(self.data[sample_function_list], 2)
        sample, same_sample = same_pair[0], same_pair[1]
        different_function = random.choice(self.function_list)
        while different_function == sample_function_list:
            different_function = random.choice(self.function_list)
        different_sample = random.choice(self.data[different_function])

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)
        
        # Pool candidates
        if self.pool_size:
            pool = self._get_pool(index)
            pool = [self._to_tensor(x) for x in pool]
            return sample, same_sample, different_sample, torch.tensor([0]), pool

        return sample, same_sample, different_sample, torch.tensor([0])
    
    
    def _get_pool(self, index):
        pool = []
        # Get the function pool that does not contain the function_name
        rest_list = self.function_list[:index] + self.function_list[index+1:]
        functions = random.sample(rest_list, self.pool_size)
        for func in functions:
            pool.append(random.choice(self.data[func]))
        return pool
        

    # @profile
    def _to_tensor(self, data: dict):
        adj = data['adj']
        feature = data['feature']
        feature = torch.FloatTensor(feature)
        # feature = torch.eye(self.feature_len)[feature]
        adj_matrix = torch.zeros([1000, 1000])
        start = torch.tensor(adj[0])
        end = torch.tensor(adj[1])
        adj_matrix[start, end] = 1

        adj_matrix = (adj_matrix + adj_matrix.T)
        adj_matrix += torch.eye(1000)

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
        adj_len = data['adj_len']
        data = data['data']
        return adj_len, feature_len, data

    def prepare_data(self):
        adj_len, feature_len, train_data = self._load_pickle_data(
            os.path.join(self.data_path, "total.pkl"))

        # Assume train_set and test_set are the same
        _, _, test_data = self._load_pickle_data(
            os.path.join(self.data_path, "test_set.pkl"))

        # total_dataset = ASTGraphDataset(data, max_adj=min(adj_len, 1000), feature_len=feature_len, exclude=self.exclude)

        self.max_length = 1000 if adj_len < 1000 else adj_len
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
    p = ASTGraphDataModule(data_path="uboot_dataset/cpg_file", pool_size=15, num_workers=2, batch_size=1)
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
        break
