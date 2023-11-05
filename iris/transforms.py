#################### IMPORTS ####################
#################################################


from typing import List, Optional, Sequence, Tuple

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

#################### CUSTOM TRANSFORM PRESETS ####################
##################################################################


class PresetTrain(torch.nn.Module):
    """
    Custom Composed Transforms for training

    Compatible with Segmentation and Classification Models

    Training augmentations:

        1. ConvertImageDtype(torch.float)
        3. RandomResize(base_size, base_size)
        4. RandomPerspective()
        5. RandomCrop(int(crop_size * base_size))
        6. StrideResize()
        7. RandomHorizontalFlip(hflip_prob)
    """

    def __init__(
        self,
        task: str = "segmentation",
        base_size: int = 320,
        crop_size: float = 0.7,
        hflip_prob: float = 0.5,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        """
        Constructor

        Arguments:
            - task: the learning task
            - base_size: the desired approximate size of resized images
            - crop_size: the desired size of cropped images as a percentage of base_size
            - hflip_prob: the probability of horizontally flipping
        """
        super(PresetTrain, self).__init__()
        self.task = task

        trans: Sequence[torch.nn.Module] = []
        trans.extend(
            [
                ConvertImageDtype(torch.float),
                RandomResize(base_size, base_size),
                RandomPerspective(),
                # RandomCrop(int(crop_size * base_size)),
                StrideResize(),
            ]
        )
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        if None not in (mean, std):
            trans.append(Normalize(mean, std))

        self.transforms = Compose(trans, task)

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        return self.transforms(image, target)


class PresetEval(torch.nn.Module):
    def __init__(
        self,
        task: str = "segmentation",
        base_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        """
        Constructor

        Arguments:
            - task: the learning task
            - base_size: the desired approximate size of resized images
        """
        super(PresetEval, self).__init__()
        self.task = task

        trans: Sequence[torch.nn.Module] = []
        trans.extend(
            [
                ConvertImageDtype(torch.float),
                RandomResize(base_size, base_size),
                StrideResize(),
            ]
        )
        if None not in (mean, std):
            trans.append(Normalize(mean, std))

        self.transforms = Compose(trans, task)

    def forward(
        self, image: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        return self.transforms(image, target)


class PresetInference(torch.nn.Module):
    def __init__(
        self,
        base_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        """
        Constructor

        Arguments:
            - task: the learning task
            - base_size: the desired approximate size of resized images
        """
        super(PresetInference, self).__init__()

        self.transforms = [
            T.ConvertImageDtype(torch.float),
            T.Resize(base_size, antialias=True),  # type: ignore
        ]

        if None not in (mean, std):
            self.transforms.append(T.Normalize(mean, std))

        self.preprocessing = torch.nn.Sequential(*self.transforms)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
        """
        return self.preprocessing(image)


#################### CUSTOM TRANSFORMS ####################
###########################################################


class Compose(torch.nn.Module):
    """
    Custom Compose torch.nn.Module

    Apply a list of custom transforms to an image and its segmentation mask target.
    """

    def __init__(
        self, transforms: Sequence[torch.nn.Module], task: str = "segmentation"
    ) -> None:
        """
        Constructor

        Arguments:
            - transforms: a list of custom torch.nn.Module transforms
        """
        super(Compose, self).__init__()
        self.transforms = transforms
        self.task = task

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(torch.nn.Module):
    """
    Custom RandomResize torch.nn.Module

    Resize an image and its segmentation mask to a random size in [min_size, max_size).
    If no max_size is provided, no resizing is applied.
    """

    def __init__(
        self,
        min_size: int = 224,
        max_size: Optional[int] = None,
    ) -> None:
        """
        Constructor

        Arguments:
            - min_size: the minimum of the range of random resized image dimensions
            - max_size: the minimum of the range of random resized image dimensions (if None, no resize is performed)
        """
        super(RandomResize, self).__init__()
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        if self.min_size <= self.max_size:
            if self.min_size == self.max_size:
                size = self.min_size
            else:
                size = int(
                    torch.randint(self.min_size, self.max_size, size=(1,)).item()
                )
            image = F.resize(image, [size, size], antialias=True)
            try:
                target = F.resize(
                    target,
                    [size, size],
                    interpolation=T.InterpolationMode.NEAREST,
                )
            except Exception as e:
                # This allows for the transform to be used on non-mask-style labels (classification probabilities)
                # NOTE: May cause problems if we integrate object detection/instance segmentation
                pass
        return image, target


class StrideResize(torch.nn.Module):
    """
    Custom stride-based resize torch.nn.Module

    Resize an image and its segmentation mask such that both height and width are multiples of stride.
    Minimally change the image aspect ratio
    """

    def __init__(self, stride: int = 32) -> None:
        """
        Constructor

        Arguments:
            - stride: the stride/layer size interval (32, 64)
        """
        super(StrideResize, self).__init__()
        self.stride = stride

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        w, h = image.shape[1:]
        desired_size = [w - (w % self.stride), h - (h % self.stride)]
        image = F.resize(image, desired_size, antialias=True)
        try:
            target = F.resize(
                target,
                desired_size,
                interpolation=T.InterpolationMode.NEAREST,
            )
        except TypeError:
            # This allows for the transform to be used on non-mask-style labels (classification probabilities)
            # NOTE: May cause problems if we integrate object detection/instance segmentation
            pass
        return image, target


class RandomCrop(torch.nn.Module):
    """
    Custom RandomCrop torch.nn.Module

    Crop an image and its segmentation mask to a specified size.
    The center of the crop is chosen randomly, where zero-padding is used if necessary.
    """

    def __init__(self, size: int = 224) -> None:
        """
        Constructor

        Arguments:
            - size: the size of the crop
        """
        super(RandomCrop, self).__init__()
        self.size = size
        self.random_crop = T.RandomCrop(size, pad_if_needed=True)

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        crop_params = self.random_crop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        try:
            target = F.crop(target, *crop_params)
        except TypeError:
            # This allows for the transform to be used on non-mask-style labels (classification probabilities)
            # NOTE: May cause problems if we integrate object detection/instance segmentation
            pass
        return image, target


class CenterCrop(torch.nn.Module):
    """
    Custom CenterCrop torch.nn.Module

    Crop an image and its segmentation mask to a specified size.
    The center of the crop is always the image center, where padding is used if necessary.
    """

    def __init__(self, size: int = 224) -> None:
        """
        Constructor

        Arguments:
            - size: the size of the crop
        """
        super(CenterCrop, self).__init__()
        self.size = size

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        image = F.center_crop(image, [self.size])
        try:
            target = F.center_crop(target, [self.size])
        except TypeError:
            # This allows for the transform to be used on non-mask-style labels (classification probabilities)
            # NOTE: May cause problems if we integrate object detection/instance segmentation
            pass
        return image, target


class RandomHorizontalFlip(torch.nn.Module):
    """
    Custom RandomHorizontalFlip torch.nn.Module

    Horizontally flip an image and its segmentation mask with probability flip_prob.
    """

    def __init__(self, flip_prob: float = 0.5) -> None:
        """
        Constructor

        Arguments:
            - flip_prob: the probability of the image being flipped
        """
        super(RandomHorizontalFlip, self).__init__()
        self.flip_prob = flip_prob

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        if torch.rand(1) < self.flip_prob:
            image = F.hflip(image)
            try:
                target = F.hflip(target)
            except TypeError:
                # This allows for the transform to be used on non-mask-style labels (classification probabilities)
                # NOTE: May cause problems if we integrate object detection/instance segmentation
                pass
        return image, target


class ConvertImageDtype(torch.nn.Module):
    """
    Custom ConvertImageDtype torch.nn.Module

    Change the datatype of an image (not applied to its segmentation mask).
    """

    def __init__(self, dtype: torch.dtype = torch.float):
        """
        Constructor

        Arguments:
            - dtype: the datatype of the image
        """
        super(ConvertImageDtype, self).__init__()
        self.dtype = dtype

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize(torch.nn.Module):
    """
    Custom Normalize torch.nn.Module

    Normalize an image (not applied to its segmentation mask).
    """

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        """
        Constructor

        Arguments:
            - mean: the mean of the un-normalized image data
            - std: the standard deviation of the un-normalized image data
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        if self.mean is not None and self.std is not None:
            image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor(torch.nn.Module):
    """
    Custom ToTensor torch.nn.Module

    ToTensor Transform (convienence)
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        super(ToTensor, self).__init__()
        self.to_tensor = T.ToTensor()

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """
        if not isinstance(image, torch.Tensor):
            # This operation converts ndarrays/PIL Images of integers [0:255] to FloatTensors [0:1]
            image = self.to_tensor(image)
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        return image, target


class RandomPerspective(torch.nn.Module):
    """
    Custom RandomPerspective torch.nn.Module

    Same behavior as the torchvision transform class of the same name.
    Compatible with segmentation masks.
    """

    def __init__(
        self,
        distortion_scale: float = 0.1,
        p: float = 0.75,
        fill: int = 0,
    ) -> None:
        """
        Constructor

        Arguments:
            - distortion_scale: the degree of distortion on [0, 1]
            - p: the probability of making a random perspective on [0, 1]
            - fill: the fill value on the edges of the resulting images
        """
        super(RandomPerspective, self).__init__()
        self.distortion_scale = distortion_scale
        self.p = p
        self.fill = fill
        self.random_perspective = T.RandomPerspective(distortion_scale, p, fill=fill)

    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Arguments:
            - image: the image (B x C x H x W)
            - target: the segmentation mask (B x 1 x H x W)

        Returns:
            - image: the transformed image (B x C x H x W)
            - target: the transformed segmentation mask (B x 1 x H x W)
        """

        perspective_params = self.random_perspective.get_params(
            image.shape[-1], image.shape[-2], distortion_scale=self.distortion_scale
        )
        image = F.perspective(image, perspective_params[0], perspective_params[1])
        try:
            target = F.perspective(
                target,
                perspective_params[0],
                perspective_params[1],
                interpolation=T.InterpolationMode.NEAREST,
            )
        except TypeError:
            # This allows for the transform to be used on non-mask-style labels (classification probabilities)
            # NOTE: May cause problems if we integrate object detection/instance segmentation
            pass
        return image, target
