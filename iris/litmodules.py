#################### IMPORTS ####################
#################################################

import sys
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MultilabelAccuracy,
    MultilabelF1Score,
)
from torchvision import models as torch_models
from torchvision import transforms as T

from iris import criteria as iris_criteria

#################### BASE CUSTOM LIGHTNING MODULE ####################
######################################################################


class IrisLitModule(LightningModule):
    """
    Base Iris LightningModule

    Aiming for compatibility/extensibility with:

    Semantic Segmentation - DONE \n
    Instance Segmentation - Use YOLO? \n
    Binary Classification - TODO \n
    Multi-class Classification - DONE \n
    Multi-label Binary-Classification - DONE \n
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        # store hyperparameters in self.hparams
        self.save_hyperparameters(cfg)

        # initialize torch network
        self.model = self._get_torch_network()

        # loss function and loss tracking
        try:
            self.criterion = getattr(torch.nn, self.hparams.criterion)()  # type: ignore
        except Exception as e:
            self.criterion = getattr(iris_criteria, self.hparams.criterion)()  # type: ignore
        self.loss_dict = {"train": [], "val": [], "test": []}

        # for ModelSummary Callback
        self.example_input_array = torch.randn(
            2,
            *self.hparams.imgsz,  # type: ignore
        )

        # for logging images
        self.arr2pil = T.ToPILImage()
        self.log_batches = []

    def forward(self, samples):
        return self.model(samples)

    #################### TRAINING ####################

    def training_step(
        self, batch: Tuple, batch_idx: int
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        samples, targets = batch
        preds = self(samples)
        loss = self._lightning_step("train", samples, targets, preds)
        return loss

    #################### VALIDATING ####################

    def validation_step(
        self, batch: Tuple, batch_idx: int
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        samples, targets = batch
        preds = self(samples)
        loss = self._lightning_step("val", samples, targets, preds)
        return loss

    #################### TESTING ####################

    def test_step(
        self, batch: Tuple, batch_idx: int
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        samples, targets = batch
        preds = self(samples)
        loss = self._lightning_step("test", samples, targets, preds)
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_images()
        self.log_batches = []

    #################### PREDICTION ####################

    def predict_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        samples = batch[0]
        preds = self(samples)
        # semantic segmentation models return dictionaries w/ "out" as the mask key
        if isinstance(preds, OrderedDict):
            return preds["out"]
        else:
            return preds

    #################### OPTIMIZERS ####################

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer  # type: ignore
        lr = self.hparams.lr  # type: ignore
        weight_decay = self.hparams.weight_decay  # type: ignore

        if optimizer in ("SGD", "Adadelta"):
            betas = self.hparams.momentum  # type: ignore
        if optimizer in (
            "Adam",
            "AdamW",
            "Adamax",
        ):
            betas = (self.hparams.momentum, 0.999)  # type: ignore

        opt = getattr(torch.optim, self.hparams.optimizer)(  # type: ignore
            self.trainer.model.parameters(), lr, betas, weight_decay=weight_decay  # type: ignore
        )

        opts = {"optimizer": opt}

        if self.hparams.scheduler == "swa":  # type: ignore
            pass
        elif self.hparams.scheduler == "linear_warmup_cosine_annealing":  # type: ignore
            #### Cosine Annealing LR with warmup (use LR finder suggestion)
            T_max = self.hparams.epochs * self.hparams.num_batches  # type: ignore
            T_0 = max(5 * self.hparams.num_batches, int(T_max / 10))  # type: ignore
            start_factor = 1e-2
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=opt, start_factor=start_factor, total_iters=T_0
            )
            annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=opt,
                T_max=T_max - T_0,
                eta_min=start_factor * self.hparams.lr,  # type: ignore
            )
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, annealing_scheduler],
                    milestones=[T_0 + 1],
                ),
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss_epoch",
                "name": "scheduler",
            }
            opts["lr_scheduler"] = scheduler_config
        else:
            #### Adaptive LR reduction
            patience = max(5, int(0.05 * self.hparams.epochs))  # type: ignore
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.25,
                    patience=patience,
                    threshold=1e-3,
                    threshold_mode="rel",
                    cooldown=patience,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss_epoch",
                "name": "scheduler",
            }
            opts["lr_scheduler"] = scheduler_config
        return opts

    #################### CUSTOM UTILITIES ####################

    def _lightning_step(
        self,
        mode: str,
        samples: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        General Lightning Step

        mode behavior:
            - "train": Loss logging
            - "val": Loss and Metric logging
            - "test" Loss, Metric, and Image logging
            - "predict" Online/Batch Inference API

        Arguments:
            - mode: the string indicating the type of lightning step to perform ["train", "val", "test", "predict"]
            - samples: the batch of samples
            - targets: the batch of targets
            - preds: the batch of predictions

        Returns:
            - loss: the loss as a tensor, often singleton
        """

        if mode in ("train", "val", "test"):
            loss = self._compute_loss(preds, targets)
            self.log(
                f"{mode}/loss",
                loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )

            if mode in ("val", "test"):
                # log any attributes of self that start with "metric_XXX", i.e. self.metric_mIoU
                self_attrs = dir(self)
                for attr in self_attrs:
                    if "metric_" in attr and attr != "_metric_attributes":
                        # wandb logger errors if we call .log() with a dict
                        metric_or_dict = getattr(self, attr)(preds, targets)
                        if isinstance(metric_or_dict, dict):
                            self.log_dict(
                                {f"{mode}/{k}": v for k, v in metric_or_dict.items()},
                                on_epoch=True,
                                sync_dist=True,
                            )
                        else:
                            self.log(
                                f"{mode}/{attr.replace('metric_', '')}",
                                metric_or_dict,
                                on_epoch=True,
                                sync_dist=True,
                            )
                if mode == "test" and len(self.log_batches) < 5:
                    # log images
                    log_batch = {
                        "samples": samples.detach().cpu(),
                        "ground_truth": targets.detach().cpu(),
                    }
                    # semantic segmentation models return dictionaries w/ "out" as the mask key
                    if isinstance(preds, OrderedDict):
                        log_batch["predictions"] = preds["out"].detach().cpu()  # type: ignore
                    else:
                        log_batch["predictions"] = preds.detach().cpu()
                    self.log_batches.append(log_batch)
            if loss.isnan().sum() == 0:
                return loss
            else:
                return

    def _get_torch_network(self) -> torch.nn.Module:
        """
        Retrieve a network from the torchvision model registry

        Returns:
            - model: a torch.nn.Module object
        """
        # torchvision models
        model = torch_models.get_model(
            self.hparams.model_arch,  # type: ignore
            weights="DEFAULT" if self.hparams.pretrained else None,  # type: ignore
        )

        # Determine the input space of the pytorch default final layer
        model_childrens = [name for name, _ in model.named_children()]
        try:
            final_layer_in_features = getattr(model, f"{model_childrens[-1]}")[
                -1
            ].in_features
        except Exception as e:
            final_layer_in_features = getattr(
                model, f"{model_childrens[-1]}"
            ).in_features
        # Replace the default output layer with a custom one
        new_output_layer = torch.nn.Linear(
            in_features=final_layer_in_features, out_features=self.hparams.num_classes  # type: ignore
        )
        try:
            getattr(model, f"{model_childrens[-1]}")[-1] = new_output_layer
        except Exception as e:
            setattr(model, model_childrens[-1], new_output_layer)

        return model

    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss, defined differently for different tasks

        Arguments:
            - outputs: the outputs of the torch.nn.Module object
            - targets: the ground truth
        Returns:
            - loss: the loss tensor (often singleton)
        """
        return self.criterion(outputs, targets)

    def _log_images(self, captions_batch=None, masks_batch=None) -> None:
        """
        Function to perform image data logging.
        """
        for i, log_batch in enumerate(self.log_batches):
            samples = [self.arr2pil(sample) for sample in log_batch["samples"]]
            if captions_batch is not None and masks_batch is not None:
                # both captions and masks
                self.logger.log_image(  # type: ignore
                    "test/image",
                    samples,
                    caption=captions_batch[i],
                    masks=masks_batch[i],
                )
            elif captions_batch is not None:
                # captions only
                self.logger.log_image("test/image", samples, caption=captions_batch[i])  # type: ignore
            elif masks_batch is not None:
                # masks only
                self.logger.log_image("test/image", samples, masks=masks_batch[i])  # type: ignore
            else:
                # just images
                self.logger.log_image("test/image", samples)  # type: ignore


#################### MULTI-CLASS CLASSIFICATION CUSTOM LIGHTNING MODULE ####################
############################################################################################


class MultiClassClassificationIrisLitModule(IrisLitModule):
    """
    Multi-class Classification Iris LightningModule
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        self.metric_acc = MulticlassAccuracy(self.hparams.num_classes)  # type: ignore
        self.metric_f1 = MulticlassF1Score(self.hparams.num_classes)  # type: ignore

    #################### CUSTOM UTILITIES ####################

    def _log_images(self):
        """
        Log images with classification targets and predictions as image captions
        """
        captions_batch = []
        for log_batch in self.log_batches:
            captions = [
                f"prediction: {self.hparams.classes[torch.argmax(pred, dim=0).item()]}, \n" + # type: ignore
                f"target: {self.hparams.classes[target.item()]}"  # type: ignore
                for pred, target in zip(
                    log_batch["predictions"], log_batch["ground_truth"]
                )
            ]
            captions_batch.append(captions)

        super()._log_images(captions_batch=captions_batch)


#################### MULTI-LABEL BINARY CLASSIFICATION LIGHTNING MODULE ####################
############################################################################################


class MultiLabelClassificationIrisLitModule(IrisLitModule):
    """
    Multi-Label Binary Classification Iris LightningModule
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        self._accuracy = MultilabelAccuracy(num_labels=self.hparams.num_classes, average=None)  # type: ignore
        self.metric_acc = lambda preds, targets: {
            f"acc_{i}": acc_i.item()
            for i, acc_i in enumerate(self._accuracy(preds, targets))
        }
        self.metric_f1 = MultilabelF1Score(num_labels=self.hparams.num_classes)  # type: ignore

    #################### CUSTOM UTILITIES ####################

    def _log_images(self):
        """
        Log images with classification targets and predictions as image captions
        """
        captions_batch = []
        for log_batch in self.log_batches:
            captions = [
                f"prediction: {[ self.hparams.classes[i] + '(' + str(round(elem, 3)) + ')' for i, elem in enumerate(pred.tolist()) if (elem > 0.5) ]}, \n" +  # type: ignore
                f"target: {[ self.hparams.classes[i] for i, elem in enumerate(target.int().tolist()) if (elem > 0.5) ]}"  # type: ignore
                for pred, target in zip(
                    log_batch["predictions"], log_batch["ground_truth"]
                )
            ]
            captions_batch.append(captions)

        super()._log_images(captions_batch=captions_batch)


#################### SEMANTIC SEGMENTATION CUSTOM LIGHTNING MODULE ####################
#######################################################################################


class SemanticSegmentationIrisLitModule(IrisLitModule):
    """
    Semantic Segmentation Iris LightningModule (inherited from Base Iris LitModule)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        if "ignore_index" not in cfg.keys():
            cfg["ignore_index"] = -100
        super().__init__(cfg)
        # loss function and loss tracking (ignoring index 0 bc it is the background)
        try:
            self.criterion = getattr(torch.nn, self.hparams.criterion)(ignore_index=self.hparams.ignore_index)  # type: ignore
        except Exception as e:
            self.criterion = getattr(iris_criteria, self.hparams.criterion)()  # type: ignore
        # Multiclass Accuracy and Multiclass Jaccard Index as performance metrics
        self._accuracy = MulticlassAccuracy(self.hparams.num_classes, ignore_index=self.hparams.ignore_index)  # type: ignore
        self._mIoU = MulticlassJaccardIndex(self.hparams.num_classes, ignore_index=self.hparams.ignore_index)  # type: ignore
        self.metric_acc = lambda preds, targets: self._accuracy(
            preds["out"], targets.squeeze(1)
        )
        self.metric_mIoU = lambda preds, targets: self._mIoU(
            preds["out"], targets.squeeze(1)
        )

    #################### CUSTOM UTILITIES ####################

    def _get_torch_network(self) -> torch.nn.Module:
        """
        Retrieve a network, either custom or torchvision

        Available Semantic Segmentation Models (self.cfg["model_arch"]):
            - RITNET (custom)
            - torchvision.models.segmentation:
                - fcn_resnet50, fcn_resnet101
                - deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101
                - lraspp_mobilenet_v3_large

        Returns:
            - model: a torch.nn.Module object
        """

        # torchvision models
        weights_backbone = "DEFAULT" if self.hparams.pretrained else None  # type: ignore
        try:
            model = torch_models.get_model(
                self.hparams.model_arch,  # type: ignore
                weights_backbone=weights_backbone,
                num_classes=self.hparams.num_classes,  # type: ignore
                aux_loss=True,
            )
        except Exception as e:
            try:
                model = torch_models.get_model(
                    self.hparams.model_arch,  # type: ignore
                    weights_backbone=weights_backbone,
                    num_classes=self.hparams.num_classes,  # type: ignore
                    aux_loss=False,
                )
            except Exception as e:
                print("Model Architecture not found")
                sys.exit(1)
        return model

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for a segmentation network

        Arguments:
            - outputs: the outputs of the torch.nn.Module object
            - targets: the ground truth
        Returns:
            - loss: the loss tensor (often singleton)
        """
        losses = {}
        for name, x in outputs.items():
            losses[name] = self.criterion(x, targets.squeeze(1))

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] + 0.5 * losses["aux"]

    def _log_images(self):
        """
        Log images with predicted segmentation masks.
        """
        masks_batch = []
        for log_batch in self.log_batches:
            masks = [
                {
                    "predictions": {
                        "mask_data": torch.argmax(pred, dim=0).numpy(),
                        # "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": torch.squeeze(target, 0).int().numpy(),
                        # "class_labels": class_labels,
                    },
                }
                for pred, target in zip(
                    log_batch["predictions"], log_batch["ground_truth"]
                )
            ]
            masks_batch.append(masks)

        super()._log_images(masks_batch=masks_batch)


#################### MODEL RETRIEVAL UTILITIES ####################
###################################################################


def get_model(
    cfg: Dict[str, Any], model_root: Optional[str] = None
) -> IrisLitModule:
    """
    Get a model, optionally by checkpoint

    Arguments:
        - cfg: the primary model/training/data config
        - model_root: (optional) the path to a model.ckpt file

    Returns:
        - lit_module: a IrisLitModule object
    """
    task_litmodule_map = {
        "segmentation": SemanticSegmentationIrisLitModule,
        "classification": MultiClassClassificationIrisLitModule,
        "multilabel": MultiLabelClassificationIrisLitModule,
    }
    if model_root is None:
        # load fresh model
        return task_litmodule_map[cfg["task"]](cfg)
    else:
        try:
            # load a checkpoint
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            return task_litmodule_map[cfg["task"]].load_from_checkpoint(
                model_root,
                map_location=map_location,
                cfg=cfg,
            )
        except Exception as e:
            print(e)
            print(
                "Checkpoint likely not found, returning requested model architecture (untrained)"
            )
            return task_litmodule_map[cfg["task"]](cfg)
