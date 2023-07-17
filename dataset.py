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
import time
import numpy as np

class ASTGraphDataset(Dataset):
    def __init__(self, data_path: str, offset: int, length: int, max_adj: int, feature_len: int) -> None:
        super().__init__()
        self.length = length
        self.offset = offset
        self.data_path = data_path
        self.data = []
        self.max_adj = max_adj
        self.feature_len = feature_len
        
        self.last_file_no = -1
        
        # The Memory is not big enough for it
        # self.data = self._prepare()

    def __len__(self):
        return self.length
    
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
    
    def _should_switch(self, index):
        if self.last_file_no * self.offset <= index < (self.last_file_no + 1) * self.offset:
            return False
        else:
            return True

    def _switch(self, index):
        file_no = index // self.offset
        with open(self.data_path[file_no], 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data, file_no

    def _handle_switch(self, index):
        if self._should_switch(index):
            data, file_no = self._switch(index)
            self.data = data
            gc.collect()
            del data
            gc.collect()
            self.last_file_no = file_no
        
        return index % self.offset
    
    # @profile
    def __getitem__(self, index):
        # impliment the sliced file read
        index = self._handle_switch(index)
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
            file_list[i] = os.path.join(os.path.abspath(data_path), file_list[i])
        sorted(file_list)
        return adj_len, feature_len, total_length, offset, file_list

    def prepare_data(self):
        adj_len, feature_len, train_total_length, offset, train_file_list = self._load_dataset_info(os.path.join(self.data_path, "total"))
        
        # Assume train_set and test_set are the same
        _, _, test_total_length, _, test_file_list = self._load_dataset_info(os.path.join(self.data_path, "test_set"))

        # total_dataset = ASTGraphDataset(data, max_adj=min(adj_len, 1000), feature_len=feature_len, exclude=self.exclude)


        self.train_set = ASTGraphDataset(data_path=train_file_list, offset=offset, length=train_total_length, max_adj=adj_len, feature_len=feature_len)
        self.val_set = ASTGraphDataset(data_path=test_file_list, offset=offset, length=test_total_length, max_adj=adj_len, feature_len=feature_len)

        self.max_length = adj_len
        self.feature_length = feature_len
        

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    

if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphDataModule(data_path="coreutil_dataset", num_workers=4)
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
