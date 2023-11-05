#################### IMPORTS ####################
#################################################


import json
from typing import Any, Dict, Optional, Sequence, Tuple

from torch import Tensor, vstack
from torchvision.io import read_image

#################### IO UTILITY FUNCTIONS ####################
##############################################################


def load_image(f: str) -> Tensor:
    """
    Convienence function for loading images from file, generally used as the loader
    argument in ImageFolder objects

    Arguments:
        - f: the file path to the image

    Returns:
        - image: a torch tensor of shape (C, H, W) and normalized to [0, 1]
    """
    image: Tensor = read_image(path=f)
    if image.shape[0] == 1:
        # masks
        return image / image.max()
    else:
        # images
        return image / 255.0


def load_sweep_config(
    cfg: Dict[str, Any], sweep_config_path: Optional[str]
) -> Dict[str, Any]:
    """
    Integrates the parameters from a weights and biases sweep into the main config

    Arguments:
        - cfg: the primary model/training/data config
        - sweep_config_path: the path to the sweep config (json)

    Retuns:
        - config: the updated primary model/training/data config
    """
    if sweep_config_path is not None:
        with open(sweep_config_path) as f:
            sweep_config = json.load(f)
        for k, v in sweep_config.items():
            cfg[k] = v
    return cfg


#################### GENERAL UTILITY FUNCTIONS ####################
###################################################################


def cat_list(images: Tuple[Tensor, ...], fill_value: int = 0) -> Tensor:
    """
    Function to generate a batch of padded tensor images

    Arguments:
        - images: a batch of torch tensor images, stored as a B-length tuple
        - fill_value: the value with which to pad tensors

    Returns:
        - batched_images: a tensor of images (B x C x H x W)
    """
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(
    batch: Tuple[Sequence[Tensor], Sequence[Tensor]]
) -> Tuple[Tensor, Tensor]:
    """
    Custom Collate Function for torch Dataloader

    Arguments:
        - batch: the raw batch (stores images and masks as tensors)

    Returns:
        - batched_imgs: a tensor of images (B x C x H x W)
        - batched_targets: a tensor of targets (B x 1 x H x W)
    """
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    try:
        # this try handles segmentation labels
        batched_targets = cat_list(targets, fill_value=0)
    except Exception as e:
        # this exception handles classification labels
        batched_targets = vstack(targets).squeeze(-1)
    return batched_imgs, batched_targets
