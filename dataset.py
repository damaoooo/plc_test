import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import random
import lightning.pytorch as pl
# import pytorch_lightning as pl
from typing import List, Union
import pickle
import os
import time
import numpy as np

class ASTGraphDataset(Dataset):
    def __init__(self, data: list, max_adj: int, feature_len: int) -> None:
        super().__init__()
        self.data = data
        self.max_adj = max_adj
        self.feature_len = feature_len
        
        # The Memory is not big enough for it
        # self.data = self._prepare()

    def __len__(self):
        return len(self.data)
    
    def _prepare(self):
        # C1 ~ C2, C1 !~ C3
        result = []
        for index in range(len(self.data)):
            sample, same_sample, different_sample = self.data[index]

            sample = self._to_tensor(sample)
            same_sample = self._to_tensor(same_sample)
            different_sample = self._to_tensor(different_sample)
            
            result.append([sample, same_sample, different_sample, torch.tensor([0])])
        return result
    
    # @profile
    def __getitem__(self, index):
        sample, same_sample, different_sample = self.data[index]

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)
        
        return sample, same_sample, different_sample, torch.tensor([0])
    
    # @profile
    def _to_tensor(self, data: dict):
        adj = data[0]
        feature = data[1]
        feature = torch.FloatTensor(feature)
        # feature = torch.eye(self.feature_len)[feature]
        adj_matrix = torch.zeros([1000, 1000])
        start = torch.tensor(adj[0])
        end = torch.tensor(adj[1])
        adj_matrix[start, end] = 1

        adj_matrix = (adj_matrix + adj_matrix.T)
        adj_matrix += torch.eye(1000)

        # For padding
        feature_padder = torch.nn.ZeroPad2d([0, self.feature_len - feature.shape[1], 0, self.max_adj - feature.shape[0]])
        feature = feature_padder(feature)[:self.max_adj][:, :self.feature_len]

        # adj_padder = torch.nn.ZeroPad2d([0, self.max_adj - adj.shape[1], 0, self.max_adj - adj.shape[0]])
        # adj = adj_padder(adj)[: self.max_adj][:, :self.max_adj]

        return feature, adj_matrix
    

class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "!pairs.pkl", batch_size: int = 32, num_workers: int = 16, exclude: list = []) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_set: Union[Dataset, None] = None
        self.val_set: Union[Dataset, None] = None
        
        self.exclude = exclude
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_length = -1
        self.feature_length = -1

    def prepare_data(self):
        with open(self.data_path, "rb") as f:
            content = pickle.load(f)
            f.close()
        data = content['data']
        adj_len = content['adj_len']
        # TODO: Should + 1?
        feature_len = content["feature_len"] 

        # total_dataset = ASTGraphDataset(data, max_adj=min(adj_len, 1000), feature_len=feature_len, exclude=self.exclude)
        total_dataset = ASTGraphDataset(data, max_adj=1000, feature_len=feature_len)
        train_data, val_data = random_split(total_dataset, [0.8, 0.2])
        # train_data, val_data = train_test_split(data, test_size=0.2)

        # self.train_set = ASTGraphDataset(train_data, max_adj=min(adj_len, 1000), feature_len=feature_len)
        # self.val_set = ASTGraphDataset(val_data, max_adj=min(adj_len, 1000), feature_len=feature_len)
        self.train_set = train_data
        self.val_set = val_data

        self.max_length = min(adj_len, 1000)
        self.feature_length = feature_len
        
        del content
        del data
        del total_dataset

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    

if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataModule(data_path="!pairs_small.pkl", num_workers=16)
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    a2 = a1
    for i in iter(train):
        idx += 1
        print(idx)
        print("Single_time:", time.time() - a2)
        a2 = time.time()
