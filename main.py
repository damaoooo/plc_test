import torch
import os
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from config import TrainConfig, read_config
from dataset import ASTGraphDataLoader
from model import BaseModel, ReGraphModel

from tensorboardX import SummaryWriter
from tqdm import tqdm

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
                      dropout=config.dropout, alpha=config.alpha, adj_len=max_length)
    model = ReGraphModel(base_model=model, pool_size=config.pool_size)
    return model


def load_optimizer(config: TrainConfig, model: ReGraphModel):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    return optimizer


def train_loop(model: ReGraphModel, train_loader: DataLoader, optimizer: optim.Optimizer, writer: SummaryWriter, bar: tqdm):
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss_basic, loss_pool, diff = output
        loss = loss_basic + loss_pool
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train/BasicLoss", loss_basic.item())
        writer.add_scalar("Train/PoolLoss", loss_pool.item())
        writer.add_scalar("Train/Diff", diff.item())
        bar.set_postfix({"BasicLoss": loss_basic.item(), "PoolLoss": loss_pool.item(), "Diff": diff.item()})
        bar.update()


def val_loop(model: ReGraphModel, val_loader: DataLoader, writer: SummaryWriter, bar: tqdm):
    model.eval()
    loss_basic_list = []
    loss_pool_list = []
    diff_list = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            output = model(data)
            loss_basic, loss_pool, diff = output.mean()
            loss_basic = loss_basic.item()
            loss_pool = loss_pool.item()
            diff = diff.item()
            loss_basic_list.append(loss_basic)
            loss_pool_list.append(loss_pool)
            diff_list.append(diff)
            bar.set_postfix({"BasicLoss": loss_basic, "PoolLoss": loss_pool, "Diff": diff})
            bar.update()
    loss_basic_mean = np.mean(loss_basic_list).item()
    loss_pool_mean = np.mean(loss_pool_list).item()
    diff_mean = np.mean(diff_list).item()
    writer.add_scalar("Val/BasicLoss", loss_basic_mean)
    writer.add_scalar("Val/PoolLoss", loss_pool_mean)
    writer.add_scalar("Val/Diff", diff_mean)


def train(config: TrainConfig):
    data_loader = load_data(config)
    model = load_model(config, data_loader.adj_len, data_loader.feature_len)
    optimizer = load_optimizer(config, model)
    writer = SummaryWriter(config.log_path)
    for epoch in range(config.max_epochs):
        train_bar = tqdm(data_loader.train_loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True)
        train_loop(model, data_loader.train_loader, optimizer, writer, train_bar)
        train_bar.close()

        val_bar = tqdm(data_loader.val_loader, desc=f"Val Epoch {epoch}", dynamic_ncols=True)
        val_loop(model, data_loader.val_loader, writer, val_bar)
        val_bar.close()

        save_model_checkpoint(model, os.path.join(config.log_path, f"model_{epoch}.pt"))


def save_model_checkpoint(model: ReGraphModel, path: str):
    torch.save(model.state_dict(), path)


def load_model_checkpoint(model: ReGraphModel, path: str):
    model.load_state_dict(torch.load(path))
    return model


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# TODO: Move to use GPU
# FP16
# GraphBolt


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    config: TrainConfig = read_config()
    random_seed = config.seed

    seed_everything(random_seed)
    train(config)

