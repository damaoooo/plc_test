import torch
import os

from torch.utils.data import DataLoader
import torch.optim as optim

from config import TrainConfig, read_config
from dataset import ASTGraphDataLoader
from model import BaseModel, ReGraphModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"


# torch.multiprocessing.set_sharing_strategy('file_system')


def load_data(config: TrainConfig):
    p = ASTGraphDataLoader(data_path=config.data_path, pool_size=config.pool_size, batch_size=config.batch_size,
                           num_workers=config.num_workers, k_fold=config.k_fold)
    return p


def load_model(config: TrainConfig, max_length: int, feature_length: int):
    model = BaseModel(in_feature=feature_length, hidden_feature=config.hidden_features,
                      out_feature=config.output_features, num_heads=config.n_heads,
                      dropout=config.n_heads, alpha=config.alpha, adj_len=max_length)
    model = ReGraphModel(base_model=model, pool_size=config.pool_size)
    return model


def load_optimizer(config: TrainConfig, model: ReGraphModel):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    return optimizer


def train_loop(model: ReGraphModel, train_loader: DataLoader, config: TrainConfig, optimizer: optim.Optimizer):
    model.train()
    # for epoch in range(config.max_epochs

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()


def val_loop(model: ReGraphModel, val_loader: DataLoader, config: TrainConfig):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            output = model(data)
            loss = output.mean()

# TODO:
# 1. Add comparison of Diff
# 2. Add basic loss and pool loss
# 3. Add Progress bar
# 4. Add Save and load
# 5. Add tensorboard support


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    config: TrainConfig = read_config()
    random_seed = config.seed
