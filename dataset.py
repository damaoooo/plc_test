
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import random
import lightning.pytorch as pl
# import pytorch_lightning as pl
from typing import List, Union
import pickle
import os
import numpy as np
import time
import dgl
from dgl.data import DGLDataset
from line_profiler import LineProfiler

os.environ["DGLBACKEND"] = "pytorch"

class ASTGraphDGLDataset(DGLDataset):
    def __init__(self, graph, name="ASTDataset", url=None, raw_dir=None, save_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
        self.graph = graph
        self.pair = {}

    def process(self):
        pass

    def __getitem__(self, idx):
        return super().__getitem__(idx)

class ASTGraphDataset(Dataset):
    def __init__(self, data: list, max_adj: int, feature_len: int, exclude: list = []) -> None:
        super().__init__()
        self.data = data
        self.exclude = exclude
        self.data = [x for x in self.data if (len(x['adj'][0]) < max_adj and x['name'] not in exclude) ]
        self.pair = self._make_pair()
        self.max_adj = max_adj
        self.feature_len = feature_len
        
    def _make_pair(self):
        pair = {}
        for data in self.data:
            if data["name"] in pair:
                pair[data['name']].append(data)
            else:
                pair[data['name']] = [data]
        return pair
    
    def _get_full(self):
        res = []
        for k in self.pair.keys():
            sample = random.choice(self.pair[k])
            fea, adj = self._to_tensor(sample)
            res.append([adj, fea, sample['name']])
        return res

    def __len__(self):
        return len(self.data)
    
    # @profile
    def __getitem__(self, index):

        sample = self.data[index]

        different_sample = random.choice(list(self.pair.keys()))
        while different_sample == sample['name']:
            different_sample = random.choice(list(self.pair.keys()))
        different_sample = random.choice(self.pair[different_sample])

        same_sample = random.choice(self.pair[sample['name']])
        # while (same_sample['opt'] == sample['opt']) and (same_sample['arch'] == sample['arch']):
        #     same_sample = random.choice(self.pair[sample['name']])

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)
        
        return sample, same_sample, different_sample, torch.tensor([0])
    
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
        feature_padder = torch.nn.ZeroPad2d([0, self.feature_len - feature.shape[1], 0, self.max_adj - feature.shape[0]])
        feature = feature_padder(feature)[:self.max_adj][:, :self.feature_len]

        # adj_padder = torch.nn.ZeroPad2d([0, self.max_adj - adj.shape[1], 0, self.max_adj - adj.shape[0]])
        # adj = adj_padder(adj)[: self.max_adj][:, :self.max_adj]

        return feature, adj_matrix
    

class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "./dataset/!total.pkl", batch_size: int = 32, num_workers: int = 16, exclude: list = []) -> None:
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
        total_dataset = ASTGraphDataset(data, max_adj=1000, feature_len=feature_len, exclude=self.exclude)
        train_data, val_data = random_split(total_dataset, [0.7, 0.3])
        # train_data, val_data = train_test_split(data, test_size=0.2)

        # self.train_set = ASTGraphDataset(train_data, max_adj=min(adj_len, 1000), feature_len=feature_len)
        # self.val_set = ASTGraphDataset(val_data, max_adj=min(adj_len, 1000), feature_len=feature_len)
        self.train_set = train_data
        self.val_set = val_data

        self.max_length = min(adj_len, 1000)
        self.feature_length = feature_len

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    

if __name__ == "__main__":
    p = ASTGraphDataModule(data_path="./c_data_json/!total.pkl", num_workers=1)
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    for i in iter(train):
        idx += 1
        print(idx)

