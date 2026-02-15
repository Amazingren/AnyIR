import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.loss_utils import FFTLoss

from utils.dataset_utils import AnyIRTrainDataset, CDD11
from net.anyir import AnyIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

if "SLURM_NTASKS" in os.environ:
    # Remove SLURM env variables to avoid issues with Lightning
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class AnyIRModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.net = AnyIR()
        
        self.loss_fn  = nn.L1Loss()
        self.aux_fn = FFTLoss(loss_weight=opt.fft_loss_weight)

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        
        aux_loss = self.aux_fn(restored,clean_patch)
        total_loss = loss + aux_loss
        
        # Logging to TensorBoard (if installed) by default
        self.log("total_loss", total_loss)
        self.log("reconstruction_loss", loss)
        self.log("aux_loss", aux_loss)

        return total_loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="AnyIR")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    model = AnyIRModel(opt)
    print(model)

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)

    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
        print(f"trainset: {trainset}")
        print(f"trainset len: {len(trainset)}")
    elif "AnyIR" in opt.trainset:
        trainset = AnyIRTrainDataset(opt)
    else:
        raise ValueError(f"Unknown trainset: {opt.trainset}")
    
    # trainset = torch.utils.data.Subset(trainset, range(256))  # for debug

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    # Optionally resume from a checkpoint
    if opt.resume_from is not None:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from)
    else:
        checkpoint_path = None

    trainer.fit(
        model=model, 
        train_dataloaders=trainloader,
        ckpt_path=checkpoint_path,
    )
    

if __name__ == '__main__':
    main()