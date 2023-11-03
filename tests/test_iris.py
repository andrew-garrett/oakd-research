#################### IMPORTS ####################
#################################################

import inspect
import os
import shutil
import sys

import torch

import iris
import iris.autocapture  # tests generated
import iris.classify
import iris.criteria  # tests generated
import iris.data
import iris.litmodules
import iris.train
import iris.transforms  # tests generated

ROOT = "./tests/"
TEST_CFG = iris.data.IrisLitDataModule.parse_config(os.path.join(ROOT, "iris.json"))
DATA_ROOT = os.path.join(ROOT, TEST_CFG["dataset_name"])

#################### AUTOCAPTURE TESTS ####################
###########################################################


def test_face_autocapture():
    """
    Face autocapture test
    """
    assert os.path.exists(os.path.join(DATA_ROOT, "images/face_test_image.jpg"))
    iris.autocapture.test_autocapture(
        mode="face", root=ROOT, dataset_name=TEST_CFG["dataset_name"], rotate=False
    )
    # Test that output is consistent
    assert os.path.exists(
        os.path.join(DATA_ROOT + "_crops", "images/face_test_image.png")
    )
    shutil.rmtree(DATA_ROOT + "_crops")


def test_eye_autocapture():
    """
    Single-eye autocapture test
    """
    assert os.path.exists(os.path.join(DATA_ROOT, "images/eye_test_image.jpg"))
    iris.autocapture.test_autocapture(
        mode="eye",
        root=ROOT,
        dataset_name=TEST_CFG["dataset_name"],
        iou_thresh=0.25,
    )
    assert os.path.exists(
        os.path.join(DATA_ROOT + "_crops", "images/eye_test_image.png")
    )
    shutil.rmtree(DATA_ROOT + "_crops")


def test_face_and_eye_autocapture():
    """
    Both-eye autocapture test
    """
    assert os.path.exists(os.path.join(DATA_ROOT, "images/face_test_image.jpg"))
    iris.autocapture.test_autocapture(
        mode="face and eye",
        root=ROOT,
        dataset_name=TEST_CFG["dataset_name"],
        rotate=False,
    )
    assert os.path.exists(
        os.path.join(DATA_ROOT + "_crops", "images/face_test_image.png")
    )
    shutil.rmtree(DATA_ROOT + "_crops")


#################### CRITERIA TESTS ####################
########################################################


def test_multiclass_criteria():
    """
    Test for ensuring criterion are differentiable and accept correct preds,targets, outputting correctly shaped items
    """
    num_classes = 4
    rng = torch.manual_seed(1234)
    class_predictions = torch.rand(
        size=(16, num_classes), generator=rng, requires_grad=True
    )
    multiclass_targets = torch.randint(0, num_classes, size=(16,), generator=rng).long()

    # test criteria on classification preds and targets
    loss = torch.nn.CrossEntropyLoss()(class_predictions, multiclass_targets)
    assert loss.requires_grad
    assert loss.shape == torch.Size([])


def test_multilabel_criteria():
    """
    Test for ensuring criterion are differentiable and accept correct preds,targets, outputting correctly shaped items
    """
    num_classes = 4
    rng = torch.manual_seed(1234)
    class_predictions = torch.rand(
        size=(16, num_classes), generator=rng, requires_grad=True
    )
    multilabel_targets = torch.randint(
        0, 2, size=(16, num_classes), generator=rng
    ).float()

    # test criteria on classification preds and targets
    loss = torch.nn.BCELoss()(class_predictions, multilabel_targets)
    assert loss.requires_grad
    assert loss.shape == torch.Size([])


def test_segmentation_criteria():
    """
    Test for ensuring criterion are differentiable and accept correct preds,targets, outputting correctly shaped items
    """
    num_classes = 4
    rng = torch.manual_seed(1234)
    mask_predictions = torch.rand(
        size=(16, num_classes, 320, 320), generator=rng, requires_grad=True
    ).float()
    mask_targets = torch.randint(
        0, num_classes, size=(16, 320, 320), generator=rng
    ).long()

    for name, cls in inspect.getmembers(sys.modules[iris.criteria.__name__]):
        if inspect.isclass(cls) and "iris.criterion" in str(cls):
            # test criteria on semantic segmentation preds and targets
            loss = cls()(mask_predictions, mask_targets)
            assert loss.requires_grad
            assert loss.shape == torch.Size([])


#################### TRANSFORMS TESTS ####################
##########################################################


def test_transforms():
    """
    Test for ensuring transforms accept correct samples,targets, outputting correctly shaped samples,targets

    Currently has testing for:
        - unlabeled data (sample, None)
        - classification data (sample, class_target)
        - semantic segmentation data (sample, mask_target)
    """
    num_classes = 4
    rng = torch.manual_seed(1234)
    sample = torch.rand(size=(3, 320, 320), generator=rng, requires_grad=True).float()
    mask_target = torch.randint(
        0, num_classes, size=(1, 320, 320), generator=rng
    ).long()
    class_target = torch.randint(0, num_classes, size=(1,), generator=rng).long()

    for name, cls in inspect.getmembers(sys.modules[iris.transforms.__name__]):
        if inspect.isclass(cls) and "iris.transforms" in str(cls):
            print(name)
            if name != "Compose":
                # Test custom transforms on unlabeled data (target = None)
                unlabeled_sample_transformed, unlabeled_target_transformed = cls()(
                    sample, None
                )
                # Test custom transforms on classification labeled data (target = class_target)
                class_sample_transformed, class_target_transformed = cls()(
                    sample, class_target
                )
                # Test custom transforms on semantic segmentation labeled data (target = mask_target)
                mask_sample_transformed, mask_target_transformed = cls()(
                    sample, mask_target
                )

                # first assert that each transformed sample is the same
                # i.e. no matter the target, the image data is still transformed consistently
                assert (
                    unlabeled_sample_transformed.shape == class_sample_transformed.shape
                    and class_sample_transformed.shape == mask_sample_transformed.shape
                )
                # then assert that each transformed sample requires gradient
                assert (
                    unlabeled_sample_transformed.requires_grad
                    and class_sample_transformed.requires_grad
                    and mask_sample_transformed.requires_grad
                )
                # then assert specific things about the behavior for each task
                # for unlabeled data
                assert unlabeled_target_transformed is None
                # for classification
                assert torch.is_tensor(class_target_transformed)
                # for semantic segmentation
                assert (
                    mask_sample_transformed.shape[1:]
                    == mask_target_transformed.shape[1:]
                )


#################### DATA TESTS ####################
####################################################


def test_unlabeled_dataset():
    """
    Test functionality of iris.data.IrisUnlabeledDataset class
    """
    dataset = iris.data.IrisUnlabeledDataset(TEST_CFG, root=ROOT)
    assert os.path.exists(os.path.join(DATA_ROOT, "dataset.json"))

    # get item functionality
    image, original_size, im_fname = dataset.__getitem__(0)
    assert isinstance(image, torch.Tensor)
    assert isinstance(original_size, torch.Size)
    assert isinstance(im_fname, str)
    os.remove(os.path.join(DATA_ROOT, "dataset.json"))


def test_multiclass_dataset():
    """
    Test functionality of iris.data.MultiClassClassificationIrisDataset class
    """
    TEST_CFG["task"] = "classification"
    dataset = iris.data.MultiClassClassificationIrisDataset(TEST_CFG, root=ROOT)
    assert os.path.exists(os.path.join(DATA_ROOT, "dataset.json"))

    # get item functionality
    image, target = dataset.__getitem__(0)
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert len(target.shape) == 0
    os.remove(os.path.join(DATA_ROOT, "dataset.json"))


def test_multilabel_dataset():
    """
    Test functionality of iris.data.MultiLabelClassificationIrisDataset class
    """
    TEST_CFG["task"] = "multilabel"
    dataset = iris.data.MultiLabelClassificationIrisDataset(TEST_CFG, root=ROOT)
    assert os.path.exists(os.path.join(DATA_ROOT, "dataset.json"))

    # get item functionality
    image, target = dataset.__getitem__(0)
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert len(target.shape) == 1
    os.remove(os.path.join(DATA_ROOT, "dataset.json"))


def test_segmentation_dataset():
    """
    Test functionality of iris.data.SemanticSegmentationIrisDataset class
    """
    TEST_CFG["task"] = "segmentation"
    dataset = iris.data.SemanticSegmentationIrisDataset(TEST_CFG, root=ROOT)
    assert os.path.exists(os.path.join(DATA_ROOT, "dataset.json"))

    # get item functionality
    image, target = dataset.__getitem__(0)
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert image.shape[1:] == target.shape[1:]
    assert target.shape[0] == 1
    os.remove(os.path.join(DATA_ROOT, "dataset.json"))
