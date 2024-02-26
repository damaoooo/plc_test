import torch
import os
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from model import PLModelForAST
from dataset import ASTGraphDataModule

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    my_dataset = ASTGraphDataModule(data_path="/home/damaoooo/Downloads/plc_test/dataset/openplc", batch_size=4, num_workers=4, pool_size=50, k_fold=5)
    my_dataset.prepare_data()

    # print(my_dataset.max_length)
    # my_model = PLModelForAST(adj_length=my_dataset.max_length, in_features=my_dataset.feature_length)

    config = {
        "in_features" : tune.choice([my_dataset.feature_length]),
        "adj_length": tune.choice([my_dataset.max_length]),
        "hidden_features": tune.choice([32, 64, 128, 256]),
        "output_features": tune.choice([16, 32, 64, 128, 256]),
        "n_heads": tune.choice([4, 6, 8, 10]),
        "dropout": tune.choice([0, 0.2, 0.4, 0.6, 0.8]),
        "alpha": tune.choice([0, 0.2, 0.4, 0.6, 0.8]),
        "lr": tune.loguniform(1e-6, 1e-3),
    }
    
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-AST", version='.')
    
    max_epoches = 10
    
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=PLModelForAST, config=config)
        .trainer(max_epochs=max_epoches, accelerator="gpu", logger=logger, precision="16-mixed")
        .fit_params(datamodule=my_dataset)
        .checkpointing(monitor="val_loss_all", save_top_k=2, mode="min")
        .build()
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss_all",
            checkpoint_score_order="min"
        ),
    )
    
    scheduler = ASHAScheduler(max_t=max_epoches, grace_period=1, reduction_factor=2)
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU":5, "GPU": 1})
    lightning_trainer = LightningTrainer(scaling_config=scaling_config, run_config=run_config)
    
    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=30,
            scheduler=scheduler
        ),
        run_config=air.RunConfig(
            name="tune-AST-ASHA",
        )
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss_all", mode="min")
    print(best_result)

