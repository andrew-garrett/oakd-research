#################### IMPORTS ####################
#################################################


import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from torch import Tensor, vstack
from torchvision.io import read_image

#################### IO UTILITY FUNCTIONS ####################
##############################################################


def save_sharpest_image(
    candidate_saves: Sequence[np.ndarray],
    mode: str = "face and eye",
    fname: str = "best.png",
) -> None:
    """
    Function to choose the best focused image from a list and save it locally

    Arguments:
        - candidate_saves: a list of opencv frames (usually the crops generated from detections)
        - mode: the autocapture mode ("face", "eye", "face and eye")
        - fname: the desired filename for the best saved crop
        - num_candidates: the number of crops to consider in sharpness optimization
    """
    save_root = os.path.dirname(fname)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if len(candidate_saves) > 0:
        best_candidate, best_focus = None, -1e6
        for candidate in candidate_saves:
            lap4_focus = np.std(cv2.Laplacian(candidate[0], cv2.CV_64F)) ** 2  # type: ignore
            if lap4_focus > best_focus:
                best_candidate, best_focus = candidate, lap4_focus
        fname_split = fname.split(".")
        if fname[0] == ".":
            fname_split = fname_split[1:]
            fname_split[0] = "." + fname_split[0]
        for i, detection in enumerate(best_candidate):  # type: ignore
            if i > 0:
                continue
            cv2.imwrite(
                fname_split[0] + "." + fname_split[1],
                detection,
            )
    else:
        print("No suitable eye crops found")


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


def enlarge_box(
    coords: Sequence[Union[float, int]], padding_factor: float = 1.5
) -> Sequence[Union[float, int]]:
    """
    Function to enlarge a detection by padding_factor

    Argument:
        - coords: a single bounding box's coordinates in [x_tl, y_tl, w, h] (can be normalized or un-normalized)

    Returns:
        - padded_coords: the enlarged bounding box coordinates in [x_tl, y_tl, w, h]
    """
    new_w, new_h = padding_factor * coords[2], padding_factor * coords[3]
    new_x = coords[0] - (new_w - coords[2]) / 2
    new_y = coords[1] - (new_h - coords[3]) / 2
    padded_coords = [new_x, new_y, new_w, new_h]
    # for un-normalized coordinates
    if isinstance(coords[0], (int, np.intc)):  # type: ignore
        padded_coords = list(map(int, padded_coords))
    return padded_coords


def box_iou(boxA: Sequence[int], boxB: Sequence[int]) -> float:
    """
    Function to compute the intersection over union of two bounding boxes

    Arguments:
        - boxA: bounding box, given by un-normalized [x_tl, y_tl, w, h] coordinates
        - boxB: bounding box, given by un-normalized [x_tl, y_tl, w, h] coordinates

    Returns:
        - iou: intersection over union of boxA and boxB
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


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
