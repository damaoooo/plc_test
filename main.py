import torch
import os
import pickle
from lightning.pytorch import Trainer
from model import PLModelForAST
from dataset import ASTGraphDataModule
from lightning.pytorch.callbacks import ModelCheckpoint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    
    my_dataset = ASTGraphDataModule(batch_size=1, num_workers=16, data_path="./c_blink/!total.pkl", exclude=['RTC_body', 'GATE_CONTROLLER_body', "RAMP_body"])
    my_dataset.prepare_data()

    print(my_dataset.max_length, my_dataset.feature_length)
    my_model = PLModelForAST(adj_length=1000, in_features=my_dataset.feature_length, lr=1e-5
                             , alpha=0.2, dropout=0.3, hidden_features=64, n_heads=6, output_features=128)

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_acc", mode="max")

    trainer = Trainer(
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=100,
        val_check_interval=0.3,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model=my_model, train_dataloaders=my_dataset, )
