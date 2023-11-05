#################### IMPORTS ####################
#################################################

import argparse
import os
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.tuner.tuning import Tuner

import iris.transforms as T_iris
from iris.data import IrisLitDataModule
from iris.export import export
from iris.litmodules import IrisLitModule, get_model
from iris.utils import load_sweep_config

#################### CUSTOM TRAINING UTILITIES ####################
###################################################################


def get_trainer(
    cfg: Dict[str, Any], debug: bool = False, n_gpus: int = 0
) -> Tuple[Optional[str], Trainer]:
    """
    Creates a trainer and optionally a checkpoint directory

    Arguments:
        - cfg: the primary model/training/data config
        - debug: indicator of whether or not to verbosely log
        - n_gpus: the number of gpus to train on (generally for distributed training)

    Returns:
        - checkpoint_root: if checkpointing enabled, the location to save checkpoints; else, None
        - trainer: a lightning trainer instance
    """
    # callbacks
    callbacks = [
        LearningRateMonitor(),
        ModelSummary(max_depth=3),
        EarlyStopping(
            monitor="val/loss_epoch", patience=max(10, int(0.1 * cfg["epochs"]))
        ),
    ]
    if cfg["scheduler"] == "swa":
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

    # logging
    run_root = f"./runs/{cfg['task']}/{cfg['dataset_name']}/{cfg['model_arch']}/"
    os.makedirs(run_root, exist_ok=True)
    logger = WandbLogger(
        name=cfg["model_arch"],
        project=f"iris-{cfg['task']}",
        save_dir=run_root,
        anonymous=True,
        log_model=True,
    )

    # parameter/environment dependent
    local_rank = os.getenv("LOCAL_RANK")
    node_rank = os.getenv("NODE_RANK")
    checkpoint_root = None
    profiler = None
    overfit_batches = 0
    if (
        local_rank is not None
        and node_rank is not None
        and local_rank == 0
        and node_rank == 0
    ) or n_gpus <= 1:
        # only save checkpoints for at most one gpu
        wandb_log_root = logger.experiment.dir[: -len("files")]
        checkpoint_root = os.path.join(wandb_log_root, "weights/")
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_root,
                monitor="val/loss_epoch",
                mode="min",
            )
        )
        # profiler
        if debug:
            profiler = PyTorchProfiler(
                dirpath=os.path.join(wandb_log_root, "logs/"), filename="perf-logs"
            )
            overfit_batches = 5

    # lightning trainer configuration
    strategy = "auto"
    accelerator = "gpu"
    devices = "auto"
    if n_gpus > 0:
        if n_gpus > 1:
            devices = n_gpus
            strategy = "ddp"
    else:
        accelerator = "cpu"

    return checkpoint_root, Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=cfg["epochs"],
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        overfit_batches=overfit_batches,
        # accumulate_grad_batches=8,
        precision=cfg["precision"],
    )


def tune(
    trainer: Trainer,
    lit_module: IrisLitModule,
    lit_datamodule: IrisLitDataModule,
    method: Literal["fit", "validate", "test", "predict"] = "fit",
):
    """
    Run batchsize finding and learning rate tuning

    Arguments:
        - trainer: a lightning trainer instance
        - lit_module: a IrisLightningModule to perform tuning
        - lit_datamodule: a IrisLightningDataModule used as the tuning dataset
    """
    # lightning tuner
    tuner = Tuner(trainer)
    optimal_batch_size = tuner.scale_batch_size(
        lit_module,
        datamodule=lit_datamodule,
        method=method,
        steps_per_trial=5,
    )
    if optimal_batch_size is not None:
        lit_datamodule.hparams.batch_size = optimal_batch_size  # type: ignore
        lit_module.hparams.batch_size = optimal_batch_size  # type: ignore

    lr_finder = tuner.lr_find(
        lit_module,
        datamodule=lit_datamodule,
        method=method,
        mode="linear",
        early_stop_threshold=10,
    )
    if lr_finder is not None:
        suggested_lr = lr_finder.suggestion()
        if suggested_lr is not None:
            lit_datamodule.hparams.lr = suggested_lr  # type: ignore
            lit_module.hparams.lr = suggested_lr  # type: ignore


#################### CUSTOM TRAINING ENGINE ####################
################################################################


def train(
    data_root: str = "./datasets/",
    debug: bool = False,
    run_tuning: bool = False,
    n_gpus: int = 0,
    sweep_config_fpath: Optional[str] = None,
) -> None:
    """
    Training Function

    1. Create a training config, dataset, and lightning modules
    2. Setup lightning callbacks, wandb logger, and profiler
    3. Tune batchsize and learning rate if necessary
    4. Train and Test the Model, with checkpointing

    Arguments:
        - cfg_fname: the filepath of the json config file
        - data_root: the root path where datasets lie
        - debug: indicator of whether or not to verbosely log
        - run_tuning: indicator of whether or not to tune the batch size and learning rate
        - n_gpus: the number of gpus to train on (generally for distributed training)
        - sweep_config_fpath: the path to a wandb sweep json config file
    """

    # create config, optionally updating it from a wandb sweep config
    cfg = IrisLitDataModule.parse_config()
    if sweep_config_fpath is not None:
        cfg = load_sweep_config(cfg, sweep_config_fpath)
    seed_everything(cfg["seed"])

    # get lightning datamodule
    lit_datamodule = IrisLitDataModule(cfg, root=data_root)

    # get fresh lightning module
    lit_module = get_model(cfg)

    # create and optionally tune the trainer
    checkpoint_root, trainer = get_trainer(cfg, debug=debug, n_gpus=n_gpus)
    if run_tuning and n_gpus <= 1:
        tune(trainer, lit_module=lit_module, lit_datamodule=lit_datamodule)

    # run training
    trainer.fit(lit_module, datamodule=lit_datamodule)

    # checkpoint_root is only created for at most one gpu
    if checkpoint_root is not None:
        # testing best model
        trainer.test(
            lit_module,
            datamodule=lit_datamodule,
            ckpt_path=os.path.join(checkpoint_root, os.listdir(checkpoint_root)[0]),
        )

        # exporting best model to onnx
        if cfg["export"]:
            # define the preprocessing module for onnx export
            preprocessing = T_iris.PresetInference(
                base_size=cfg["imgsz"][1],
                mean=lit_datamodule.fit_dataset.channel_means.tolist() if cfg["normalize"] else None,  # type: ignore
                std=lit_datamodule.fit_dataset.channel_means.tolist() if cfg["normalize"] else None,  # type: ignore
            )
            # export the model
            export(
                os.path.join(checkpoint_root, os.listdir(checkpoint_root)[0]),
                preprocessing=preprocessing,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Training script for iris",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="./datasets/",
        type=str,
        help="The root where the dataset folder is located",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (verbose logging, profiling, overfitted training)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune the model for batch size, learning rate",
    )
    parser.add_argument(
        "--n-gpus",
        default=torch.cuda.device_count(),
        type=int,
        help="Number of GPUs, 0 means cpu, 1 means single gpu, >1 means distributed",
    )
    parser.add_argument(
        "--sweep",
        help="The path to a wandb sweep config file",
        type=str,
    )
    ARGS = parser.parse_args()

    # run training
    wandb.login(key=os.getenv("WANDB_API_KEY"), force=True)
    train(
        data_root=ARGS.data_root,
        debug=ARGS.debug,
        run_tuning=ARGS.tune,
        n_gpus=ARGS.n_gpus,
        sweep_config_fpath=ARGS.sweep,
    )
