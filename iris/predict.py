#################### IMPORTS ####################
#################################################

import argparse
import base64
import json
import logging
import os
import sys
from typing import Any, List, Literal, Optional

import torch
from lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from iris.data import IrisLitDataModule, IrisUnlabeledDataset
from iris.litmodules import IrisLitModule, get_model

#################### CUSTOM INFERENCE UTILITIES ####################
####################################################################


class IrisPredictionWriter(BasePredictionWriter):
    """
    Pytorch Lightning PredictionWriter Callback


    write_interval = "batch":
        - (Experimental) Online-Inference Functionality
        - writes preprocessed samples to .jpg and (optionally) model predictions to .png files in the output_root directory

    write_interval = "epoch":
        - Batch-Inference Functionality
        - writes model predictions and batch indices to .pt files in the output_root directory
        - files are numbered by the device-rank, mainly used for distributed setting
    """

    def __init__(
        self,
        output_root: str = "./",
        write_interval: Literal[
            "batch", "epoch", "batch_and_epoch"
        ] = "batch_and_epoch",
    ):
        """
        Constructor

        Arguments:
            - output_root: the directory to write predictions to disk.
            - write_interval: events on which to write predictions, one of ["batch", "epoch", "batch_and_epoch"]
        """
        super().__init__(write_interval)
        self.output_root = output_root
        # ensure that the output location exists
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Write predictions to disk after passing a single batch.  Saves
        preprocessed image and predictions.
        """
        samples = batch[0]
        for sample, sample_fname, ind, pred in zip(
            samples, batch[2], batch_indices, prediction  # type: ignore
        ):
            # save the preprocessed sample
            preprocessed_sample_fname = os.path.join(
                self.output_root, os.path.basename(sample_fname)
            )
            save_image(sample, preprocessed_sample_fname)

            # save a mask if applicable
            if pl_module.hparams.task == "segmentation":  # type: ignore
                _mask = torch.nn.functional.softmax(pred, dim=0).argmax(dim=0) / 3
                save_image(_mask, preprocessed_sample_fname.replace(".jpg", ".png"))

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ) -> None:
        """
        Write predictions to disk after passing through an entire dataset.  Saves
        predictions and corresponding batch indices for each GPU.
        """
        # this will create N (num processes) files in `output_root` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_root, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_root, f"batch_indices_{trainer.global_rank}.pt"),
        )


def get_predictor(output_root: str = "./", n_gpus: int = 0) -> Trainer:
    """
    Creates a trainer and optionally a checkpoint directory

    Arguments:
        - output_root: the directory to write predictions to disk.
        - n_gpus: the number of gpus to train on (generally for distributed training)

    Returns:
        - predictor: a lightning trainer instance, used for prediction
    """
    # callbacks
    callbacks = IrisPredictionWriter(output_root=output_root)

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

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        callbacks=callbacks,  # type: ignore
        # precision="16-mixed",
    )


#################### CUSTOM INFERENCE ENGINE ####################
#################################################################


def predict(
    lit_module: IrisLitModule,
    root: str = "./datasets/",
    n_gpus: int = 0,
) -> Optional[List[Any]]:
    """
    Predict/Inference Function

    1. Create a config and unlabeled dataset
    2. Load lightning module from checkpoint
    3. Setup lightning callbacks, wandb logger, and profiler
    4. Create predictor and predict on a unlabeled dataset

    Arguments:
        - lit_module: a IrisLightningModule to perform prediction
        - root: the root where the dataset folder is located
        - n_gpus: the number of gpus to train on (generally for distributed training)
    """

    # config, dataset, and lit modules
    seed_everything(cfg["seed"])
    unlabeled_dataset = IrisUnlabeledDataset(cfg, root=root, stage="predict")
    unlabeled_dataloader = DataLoader(unlabeled_dataset, num_workers=8)
    # set number of batches
    cfg["num_batches"] = len(unlabeled_dataloader) // cfg["batch_size"]
    predictor = get_predictor(
        output_root=os.path.join(unlabeled_dataset.root, "preds"), n_gpus=n_gpus
    )

    # inferencing with the best model
    preds = predictor.predict(lit_module, dataloaders=unlabeled_dataloader)  # type: ignore
    if preds is not None:
        preds = [pred.tolist() for pred in preds]  # type: ignore
        return preds


####################  AZURE INFERENCE ONLINE ENDPOINT UTILITIES ####################
####################################################################################


def init(
    model_root: Optional[str] = f"{os.getenv('AZUREML_MODEL_DIR')}",
    model_arch: Optional[str] = None,
    model_id: Optional[str] = None,
    model_alias: Optional[str] = "best",
    data_root: str = "/var/azureml-app/iris/datasets/",
    config_fname: str = "iris.json",
):
    """
    Initializes Model for Inference, called when an Azure Endpoint is activated.

    Arguments:
        - model_root: (optional) the directory where a checkpoint (model.ckpt) is stored
        - model_arch: (optional) the model architecture
        - model_id: (optional) the wandb model id
        - model_alias: (optional) the wandb model alias,
        - data_root: the directory where a dataset is stored or the path to a json file,
        - config_fname: the path to the desired config file,
    """
    global model
    global cfg
    cfg = IrisLitDataModule.parse_config(config_fname)
    # update the number of classes using ignore indices
    if "num_classes" in cfg.keys() and "ignore_index" in cfg.keys():
        ignore_index = cfg["ignore_index"]
        if type(ignore_index) != list:
            ignore_index = [ignore_index]
        ignore_index = [i for i in ignore_index if (i >= 0 and i <= cfg["num_classes"])]
        cfg["num_classes"] -= len(ignore_index)
    # if the model_root arg is not provided, we build the path
    if model_root is None:
        # if either the model_arch or model_id arg's are not provided, we use the default model for the task
        if None in [model_arch, model_id]:
            default_model_paths = {
                "segmentation": "fcn_resnet50/qixnu203/best/model.ckpt",
                "classification": "resnet50/f172idjz/best/model.ckpt",
                "multilabel": "resnet50/ytm89y60/best/model.ckpt",
            }
            model_root = f"../../models/{default_model_paths[cfg['task']]}"
        else:
            model_root = (
                f"../../models/{model_arch}/{model_id}/{model_alias}/model.ckpt"
            )
    model = get_model(cfg, model_root)
    if data_root is not None:
        if not os.path.exists(f"{data_root}{cfg['dataset_name']}/images/"):
            os.makedirs(f"{data_root}{cfg['dataset_name']}/images/")

    return model


def process_request(raw_data: str, data_root: str):
    """
    Process a request, given as a json string, by saving its content's as images in a folder

    Arguments:
        - raw_data: the json-formatted string payload
        - data_root: the directory to save images stored on the payload
    """
    data = json.loads(raw_data)["data"]
    for i, imstr in enumerate(data):
        filename = f"{data_root}iris-inference/images/sample-{str(i).zfill(3)}.jpg"
        imgdata = base64.b64decode(imstr)

        with open(filename, "wb") as f:
            f.write(imgdata)


def run(
    raw_data: str,
    data_root: str = "/var/azureml-app/iris/datasets/",
    n_gpus: int = torch.cuda.device_count(),
):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.

    Arguments:
        - raw_data: the json-formatted string payload
        - data_root: the directory to save images stored on the payload
        - n_gpus: the number of GPUs to use
    """
    logging.info("request received")
    process_request(raw_data, data_root)
    if model is not None:
        preds = predict(model, root=data_root, n_gpus=n_gpus)
        logging.info("Request processed")
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Inference script for iris",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=str,
        help="The directory for the model checkpoint, expected to be model.ckpt. \
            if not specifed, must provide a valid pairing of [--model-arch, --model-id] arguments",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        help="The desired model architecture",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="The wandb run ID for the model checkpoint",
    )
    parser.add_argument(
        "--model-alias",
        default="best",
        type=str,
        choices=["best", "latest", "v0"],
        help="The wandb artifact alias for the model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        default="./tmp/",
        type=str,
        help="The root where the dataset folder is located",
    )
    parser.add_argument(
        "--n-gpus",
        default=torch.cuda.device_count(),
        type=int,
        help="Number of GPUs, 0 means cpu, 1 means single gpu, >1 means distributed",
    )
    ARGS = parser.parse_args()

    # if the data_root arg is provided as a json file, we assume that it is a request
    if ".json" in os.path.basename(ARGS.data_root):
        # get the model and create write directory
        model = init(
            model_root=ARGS.model_root,
            model_arch=ARGS.model_arch,
            model_id=ARGS.model_id,
            model_alias=ARGS.model_alias,
            data_root="./tmp/",
        )
        # run inference
        with open(ARGS.data_root, "r") as f:
            run(
                json.dumps(json.load(f)),
                data_root="./tmp/",
                n_gpus=ARGS.n_gpus,
            )
    else:
        # get the model and create write directory
        model = init(
            model_root=ARGS.model_root,
            model_arch=ARGS.model_arch,
            model_id=ARGS.model_id,
            model_alias=ARGS.model_alias,
            data_root=ARGS.data_root,
        )
        # run inference
        predict(model, root=ARGS.data_root, n_gpus=ARGS.n_gpus)
