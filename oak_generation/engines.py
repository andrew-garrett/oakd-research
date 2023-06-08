# #################################################
# #################### IMPORTS ####################
# #################################################


import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger

from oak_generation.models import custom_models
from utils import initialize_wandb, prepareTorchDataset

#########################################################
#################### TRAINING ENGINE ####################
#########################################################

def base_engine(cfg_fname):

    cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

    trainloader, testloader = prepareTorchDataset(cfg_wandb)

    # Set random seed for reproducability
    pl.seed_everything(cfg_wandb.seed)

    # Initialize model
    try:
        model = getattr(
            custom_models, 
            cfg_dict["model_arch"].upper(),
        )(cfg_dict)
    except Exception as e:
        print(e)
        sys.exit(1)

    logger = WandbLogger(
        name=cfg_dict["name"],
        project=cfg_dict["project"],
    )

    run_dir = f"./runs/{cfg_dict['task']}/{cfg_dict['dataset_name']}/{cfg_dict['model_arch']}/{cfg_dict['name']}/weights/"
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(dirpath=run_dir, monitor="train/g_loss_epoch", mode="max"),
        ModelSummary(max_depth=3)
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg_wandb.epochs,
        logger=logger,
        # check_val_every_n_epoch=5,
        callbacks=callbacks,
    )
    trainer.fit(model, trainloader) #, testloader)