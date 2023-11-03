#################### IMPORTS ####################
#################################################

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#################### CUSTOM LOSS FUNCTIONS ####################
###############################################################


class FocalLoss(nn.Module):
    """
    Pytorch nn.Module implementation of Multi-class Focal Loss
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2,
        ignore_index: int = 0,
    ) -> None:
        """
        Constructor

        Arguments:
            - alpha: the class weights, tensor of size (C,)
            - gamma: the smoothing value
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward call of focal loss

        Arguments:
            - outputs: the unnormalized model outputs, tensor of size (B, C, H, W)
            - targets: the targets, tensor of size (B, H, W)
        Return:
            - loss: the focal loss, tensor of size (1,)
        """
        logpt = -self.ce_loss(outputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Pytorch nn.Module implementation of Generalized Dice Loss
    """

    def __init__(
        self, alpha: Optional[torch.Tensor] = None, smooth: float = 1e-10
    ) -> None:
        """
        Constructor

        Arguments:
            - alpha: the class weights, tensor of size (C,)
            - smooth: the smoothing value
        """
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        if alpha is not None:
            self.alpha = alpha / torch.sum(alpha)  # Normalized weight
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward call of diceloss

        Arguments:
            - outputs: the unnormalized model outputs, tensor of size (B, C, H, W)
            - targets: the targets, tensor of size (B, H, W)
        Return:
            - loss: the dice loss, tensor of size (1,)
        """
        N, C = outputs.shape[:2]
        # one-hot encode targets
        targets = torch.moveaxis(F.one_hot(targets, C), -1, 1).type(
            torch.float32
        )  # (N, 1, H, W) -> (N, C, H, W)

        # take softmax of outputs
        outputs = F.softmax(outputs, dim=1).type(
            torch.float32
        )  # (N, C, H, W) -> (N, C, H, W)

        # compute weight factor = sum(outputs * targets, dim=(2, 3))
        if self.alpha is None:
            w: torch.Tensor = 1 / (
                (torch.einsum("bkwh->bk", targets).type(torch.float32) + self.smooth)
                ** 2
            )  # (N, C, H, W)
        else:
            w: torch.Tensor = self.alpha

        # compute intersection = sum(outputs * targets, dim=(2, 3))
        intersection: torch.Tensor = w * torch.einsum(
            "bkwh,bkwh->bk", outputs, targets
        )  # (N, C)

        # compute union = sum(outputs, dim=(2, 3)) + sum(targets, dim=(2, 3))
        union: torch.Tensor = w * (
            torch.einsum("bkwh->bk", outputs) + torch.einsum("bkwh->bk", targets)
        )  # (N, C)

        # compute dice loss = 1 - ((2*intersection) / (union))
        dice: torch.Tensor = (
            2
            * (torch.einsum("bk->b", intersection) + self.smooth)
            / (torch.einsum("bk->b", union) + self.smooth)
        )
        loss = 1 - dice.mean()

        return loss
