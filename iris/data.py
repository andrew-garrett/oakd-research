#################### IMPORTS ####################
#################################################


import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Tuple

import pkg_resources
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

import iris.transforms as T_iris
from iris.utils import collate_fn, load_image

#################### CUSTOM UNLABELED DATASET CLASS ####################
########################################################################


class IrisUnlabeledDataset(VisionDataset):
    """
    Custom torch Dataset of unlabeled image data

    Attributes:
        - cfg: the primary model/training/data config
        - root: the folder where the images are located (i.e. root/*.png)
    """

    def __init__(
        self, cfg: Dict[str, Any], root: str = "./datasets/", stage: str = "fit"
    ) -> None:
        """
        Arguments:
            - cfg: the primary model/training/data config
            - root: the root where the dataset folder is located
            - stage: the current stage, one of ["fit", "val", "test"]
        """
        self.cfg = cfg
        super().__init__(os.path.join(root, cfg["dataset_name"]))

        if os.path.exists(self.root):
            # get the samples
            self.data = list(
                os.path.join(self.root, "images", f)
                for f in os.listdir(os.path.join(self.root, "images"))
            )

            # compute channel_means and channel_stds
            self.transforms = T_iris.PresetEval(
                task=self.cfg["task"],
                base_size=self.cfg["imgsz"][1],
            )
            self.get_dataset_params()

            # define dataset splits
            split = [int(0.7 * len(self.data)), int(0.25 * len(self.data))]
            split.append(len(self.data) - split[0] - split[1])
            train_inds, val_inds, test_inds = random_split(
                range(len(self.data)),  # type: ignore
                split,
                generator=torch.Generator().manual_seed(self.cfg["seed"]),
            )

            # select data subset and redefine transforms according to stage
            if stage == "fit":
                self.data = sorted([self.data[i] for i in train_inds])
                self.transforms = T_iris.PresetTrain(
                    task=self.cfg["task"],
                    base_size=self.cfg["imgsz"][1],
                    mean=self.channel_means if self.cfg["normalize"] else None,  # type: ignore
                    std=self.channel_stds if self.cfg["normalize"] else None,  # type: ignore
                )
            else:
                if stage == "val":
                    self.data = sorted([self.data[i] for i in val_inds])
                elif stage == "test":
                    self.data = sorted([self.data[i] for i in test_inds])
                else:  # for the case when stage == "predict", do nothing
                    pass
                self.transforms = T_iris.PresetEval(
                    task=self.cfg["task"],
                    base_size=self.cfg["imgsz"][1],
                    mean=self.channel_means if self.cfg["normalize"] else None,  # type: ignore
                    std=self.channel_stds if self.cfg["normalize"] else None,  # type: ignore
                )
        else:
            print("Unlabeled Dataset not found")
            sys.exit(1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Size, str]:
        """
        Arguments:
            - index: the integer specifying the batch index in the dataset

        Returns:
            - sample: the image (3 x H x W)
            - original_size: the original size of the untransformed sample
            - mask_fname: the desired full filename of the mask to be saved
        """
        im_fname = self.data[index]
        sample = read_image(im_fname)
        original_size = sample.shape[1:]
        return self.transforms(sample, None)[0], original_size, im_fname

    def __len__(self) -> int:
        """
        Returns:
            - n_samples: total number of samples in the dataset
        """
        return len(self.data)

    def get_dataset_params(self):
        """
        Determine the channel-wise mean and std

        Saves parameters to a json file at self.root/dataset.json

        If the parameters have already been computed, parameters are fetched from a json file at self.root/dataset.json
        """
        if os.path.exists(os.path.join(self.root, "dataset.json")):
            with open(os.path.join(self.root, "dataset.json"), "r") as f:
                dataset_params = json.load(f)
                self.channel_means = torch.as_tensor(dataset_params["channel_means"])
                self.channel_stds = torch.as_tensor(dataset_params["channel_stds"])
        else:
            channel_means, channel_stds = torch.zeros(3), torch.zeros(3)
            for index in range(len(self)):
                # for normalization
                sample = load_image(self.data[index])
                channel_means += sample.mean(dim=(1, 2))
                channel_stds += sample.std(dim=(1, 2))

            self.channel_means = channel_means / len(self)
            self.channel_stds = channel_stds / len(self)

            # write to a file
            with open(os.path.join(self.root, "dataset.json"), "w") as f:
                json.dump(
                    {
                        "channel_means": self.channel_means.tolist(),
                        "channel_stds": self.channel_stds.tolist(),
                    },
                    f,
                    indent="\t",
                )

        print(f"Dataset has channel-wise means of {self.channel_means}")
        print(f"Dataset has channel-wise stds of {self.channel_stds}")


#################### CUSTOM MULTICLASS CLASSIFICATION DATASET CLASS ####################
########################################################################################


class MultiClassClassificationIrisDataset(IrisUnlabeledDataset):
    """
    Custom Dataset of multiclass classification labeled database

    Attributes:
        - cfg: the primary model/training/data config
        - root: the folder where the images are located (i.e. root/*.png)
    """

    def __init__(
        self, cfg: Dict[str, Any], root: str = "./datasets/", stage: str = "fit"
    ) -> None:
        super().__init__(cfg, root, stage)
        # get all labels
        with open(os.path.join(self.root, "diagnoses.csv"), "r") as f:
            # preallocate empty array of targets to ensure that sample and target indices match
            targets = [None for _ in range(len(self.data))]

            # get the class mappings
            classes = f.readline().replace("\n", "").split(", ")[1:]
            # handle ignore indices
            if "ignore_index" not in self.cfg.keys():
                ignore_index = []
            else:
                ignore_index = self.cfg["ignore_index"]
                if type(ignore_index) != list:
                    ignore_index = [ignore_index]
            keep_mask = [
                True if i not in ignore_index else False for i in range(len(classes))
            ]
            # handle class mapping with ignore indices
            cfg["classes"] = [c for keep, c in zip(keep_mask, classes) if keep]
            cfg["num_classes"] = len(cfg["classes"])

            for line in f.readlines():
                split_line = line.split(", ")
                # determine the fname of the corresponding sample to the current target
                sample_fname = os.path.join(self.root, "images", split_line.pop(0))
                # determine the index of the corresponding sample to the current target
                if sample_fname in self.data:
                    sample_index = self.data.index(sample_fname)
                    targets[sample_index] = torch.tensor(list(map(int, split_line)))[keep_mask].argmax()  # type: ignore
            if None in targets:
                self.data = [
                    self.data[i] for i in range(len(targets)) if targets[i] is not None
                ]
                self.targets = [
                    targets[i] for i in range(len(targets)) if targets[i] is not None
                ]
            else:
                self.targets = targets

            # Ensures that targets is completely and properly populated
            assert None not in self.targets
            assert len(self.data) == len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            - index: the integer specifying the batch index in the dataset

        Returns:
            - sample: the image (3 x H x W)
            - target: the multiclass target
        """
        sample = load_image(self.data[index])
        try:
            target = self.targets[index]
        except Exception as e:
            target = None
        return self.transforms(sample, target)


#################### CUSTOM MULTILABEL CLASSIFICATION DATASET CLASS ####################
########################################################################################


class MultiLabelClassificationIrisDataset(IrisUnlabeledDataset):
    """
    Custom Dataset of multilabel classification labeled database

    Attributes:
        - cfg: the primary model/training/data config
        - root: the folder where the images are located (i.e. root/*.png)
    """

    def __init__(
        self, cfg: Dict[str, Any], root: str = "./datasets/", stage: str = "fit"
    ) -> None:
        super().__init__(cfg, root, stage)
        # get all labels
        with open(os.path.join(self.root, "exam-features.csv"), "r") as f:
            # preallocate empty array of targets to ensure that sample and target indices match
            targets = [None for _ in range(len(self.data))]

            # get the class mappings
            classes = f.readline().replace("\n", "").split(", ")[1:]
            # handle ignore indices
            if "ignore_index" not in self.cfg.keys():
                ignore_index = []
            else:
                ignore_index = self.cfg["ignore_index"]
                if type(ignore_index) != list:
                    ignore_index = [ignore_index]
            keep_mask = [
                True if i not in ignore_index else False for i in range(len(classes))
            ]
            # handle class mapping with ignore indices
            cfg["classes"] = [c for keep, c in zip(keep_mask, classes) if keep]
            cfg["num_classes"] = len(cfg["classes"])

            for line in f.readlines():
                split_line = line.split(", ")
                # determine the fname of the corresponding sample to the current target
                sample_fname = os.path.join(self.root, "images", split_line.pop(0))
                # determine the index of the corresponding sample to the current target
                if sample_fname in self.data:
                    sample_index = self.data.index(sample_fname)
                    targets[sample_index] = torch.tensor(list(map(int, split_line)))[keep_mask]  # type: ignore
            if None in targets:
                self.data = [
                    self.data[i] for i in range(len(targets)) if targets[i] is not None
                ]
                self.targets = [
                    targets[i] for i in range(len(targets)) if targets[i] is not None
                ]
            else:
                self.targets = targets
            # Ensures that targets is completely and properly populated
            assert None not in self.targets
            assert len(self.data) == len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            - index: the integer specifying the batch index in the dataset

        Returns:
            - sample: the image (3 x H x W)
            - target: the multilabel target
        """
        sample = load_image(self.data[index])
        try:
            target = self.targets[index].float()  # type: ignore
        except Exception as e:
            target = None
        return self.transforms(sample, target)


#################### CUSTOM SEMANTIC SEGMENTATION DATASET CLASS ####################
####################################################################################


class SemanticSegmentationIrisDataset(IrisUnlabeledDataset):
    """
    Custom torch Dataset of semantic segmentation labeled database

    Attributes:
        - cfg: the primary model/training/data config
        - root: the folder where the images are located (i.e. root/*.png)
    """

    def __init__(
        self, cfg: Dict[str, Any], root: str = "./datasets/", stage: str = "fit"
    ) -> None:
        super().__init__(cfg, root, stage)
        targets = [None for _ in range(len(self.data))]
        for i, sample in enumerate(self.data):
            # determine the fname of the corresponding sample to the current target
            label_fname = sample.replace("images", "labels").replace(".jpg", ".png")
            if os.path.exists(label_fname):
                targets[i] = label_fname  # type: ignore
        if None in targets:
            self.data = [
                self.data[i] for i in range(len(targets)) if targets[i] is not None
            ]
            self.targets = [
                targets[i] for i in range(len(targets)) if targets[i] is not None
            ]
        else:
            self.targets = targets

        # Ensures that targets is completely and properly populated
        assert None not in self.targets
        assert len(self.data) == len(self.targets)

        # determine the classes in the dataset
        self.find_classes()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            - index: the integer specifying the batch index in the dataset

        Returns:
            - sample: the image (3 x H x W)
            - target: the mask target (1 x H x W)
        """
        sample = read_image(self.data[index])
        try:
            target = load_image(self.targets[index])  # type: ignore
            target = ((len(self.pixel2class) - 1) * target).long()
        except Exception as e:
            target = None
        return self.transforms(sample, target)

    def find_classes(self):
        """
        Determine the the pixel2class mapping.

        Saves parameters to a json file at self.root/dataset.json

        If the parameters have already been computed, parameters are fetched from a json file at self.root/dataset.json
        """
        if os.path.exists(os.path.join(self.root, "dataset.json")):
            with open(os.path.join(self.root, "dataset.json"), "r") as f:
                dataset_params = json.load(f)
                if "pixel2class" in dataset_params.keys():
                    self.pixel2class = dataset_params["pixel2class"]
                    print(f"Dataset has pixel2class mapping of {self.pixel2class}")
                    return

        pixel2class = []
        for index in range(len(self)):
            if len(pixel2class) == self.cfg["num_classes"]:
                break
            # for pixel2class mapping
            target = read_image(self.targets[index])  # type: ignore
            for c_i in torch.unique(target):
                if c_i.item() not in pixel2class:
                    pixel2class.append(int(c_i.item()))

        self.pixel2class = sorted(pixel2class)
        # get the already set params and add the pixel2class data.
        with open(os.path.join(self.root, "dataset.json"), "r") as f:
            dataset_params = json.load(f)
        dataset_params["pixel2class"] = self.pixel2class
        with open(os.path.join(self.root, "dataset.json"), "w") as f:
            json.dump(
                dataset_params,
                f,
                indent="\t",
            )
        print(f"Dataset has pixel2class mapping of {self.pixel2class}")


#################### CUSTOM LIGHTNING DATAMODULE CLASS ####################
###########################################################################


class IrisLitDataModule(LightningDataModule):
    """
    Custom Iris LightningDataModule for a Iris torch Dataset
    """

    def __init__(self, cfg: Dict[str, Any], root: str = "./datasets/") -> None:
        """
        Arguments:
            - cfg: the primary model/training/data config
            - root: the root where the dataset folder is located
        """
        super().__init__()

        # define the dataset splits
        self.fit_dataset = get_dataset(cfg, root, "fit")
        self.val_dataset = get_dataset(deepcopy(cfg), root, "val")
        self.test_dataset = get_dataset(deepcopy(cfg), root, "test")

        # save the hyperparmeters after creating the training dataset because
        # we require the "num_batches" hyperparameter, which is set in training dataset creation
        self.save_hyperparameters(self.fit_dataset.cfg)  # type: ignore
        self.total_num_workers = os.cpu_count() if os.cpu_count() is not None else 1

    @staticmethod
    def parse_config(config_fname: str = "iris.json") -> Dict[str, Any]:
        """
        Creates the iris config dictionary

        At minimum, iris.json specifies the desired learning task:
            - "classification" for single-label, multi-class classification
            - "multilabel" for multi-label, binary classification
            - "segmentation" for semantic segmentation
        If any other parameters are passed to iris.json, these parameters override
        those set in the task-specific config.

        Returns:
            - config: a dictionary of model hyperparameters
        """
        try:
            # retrieve the main config
            with open(config_fname, "r") as f:
                iris_config = json.load(f)
            # retrieve the task specific config
            task_config_fname = pkg_resources.resource_filename(
                "iris", os.path.join("configs", f"{iris_config['task']}.json")
            )
            with open(task_config_fname, "r") as f:
                task_config = json.load(f)
                for k, v in task_config.items():
                    if k not in iris_config.keys():
                        iris_config[k] = v
            return iris_config
        except Exception as e:
            print(e)
            print(
                f"Config file not found: \n path: {os.getcwd()} \n filename: iris.json"
            )
            sys.exit(1)

    def _create_dataloader(self, stage: str = "fit") -> DataLoader:
        """
        Generalized Dataloader Creation.

        Parameters:
            - stage: ["fit", "val", "test"]

        Returns:
            - dataloader: A torch.utils.data.DataLoader object for the specifed split
        """
        # Assumes that total number of workers is at least 8
        if stage == "fit":
            num_workers = int(self.total_num_workers * 1.0)  # type: ignore
        elif stage == "val":
            num_workers = int(self.total_num_workers * 0.75)  # type: ignore
        else:
            num_workers = int(self.total_num_workers * 0.5)  # type: ignore

        return DataLoader(
            getattr(self, f"{stage}_dataset"),
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=(stage == "fit"),
            num_workers=max(1, num_workers),
            generator=torch.Generator().manual_seed(self.hparams.seed),  # type: ignore
            pin_memory=True,
            collate_fn=collate_fn,  # type: ignore
        )

    def train_dataloader(self):
        return self._create_dataloader("fit")

    def val_dataloader(self):
        return self._create_dataloader("val")

    def test_dataloader(self):
        return self._create_dataloader("test")


#################### DATASET RETRIEVAL UTILITIES ####################
#####################################################################


def get_dataset(
    cfg: Dict[str, Any], data_root: str = "./datasets/", stage: str = "fit"
) -> Dataset:
    """
    Prepare a torch Dataset object

    Currently supports:
        - UBIPR Semantic Segmentation datasets
        - Corneacare Custom Dataset

    Arguments:
        - cfg: the primary model/training/data config
        - data_root: the root path where datasets lie

    Returns:
        - dataset: a torch Dataset
    """
    task_litmodule_map = {
        "segmentation": SemanticSegmentationIrisDataset,
        "classification": MultiClassClassificationIrisDataset,
        "multilabel": MultiLabelClassificationIrisDataset,
    }
    try:
        # load fresh dataset
        dataset = task_litmodule_map[cfg["task"]](cfg, data_root, stage)
        cfg["num_batches"] = len(dataset) // cfg["batch_size"]
        dataset.__setattr__("cfg", cfg)
        return dataset
    except Exception as e:
        print(e)
        print("Lightning DataModule not supported for this task")
        sys.exit(1)
