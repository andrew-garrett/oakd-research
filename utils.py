#################################################
#################### IMPORTS ####################
#################################################


import json
import math
import os
import shutil
import ssl
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from roboflow import Roboflow
from tqdm import tqdm

import wandb

###########################################################
#################### GENERAL UTILITIES ####################
###########################################################


def initialize_wandb(cfg_fname):
    """
    Function to initialize wandb experiment, configured by the specified config file

    Arguments
    - cfg_fname (string): specifies the file location of model_cfg.json

    Returns:
    - model_cfg (dict): dictionary storing data from model_cfg.json
    - config (wandb.Config): config object for wandb experiment
    """
    with open(cfg_fname, "r") as f:
        model_cfg = json.load(f)
    model_cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg["project"] = f"oakd-research-{model_cfg['task'].replace('_', '-')}"
    model_cfg["lr_steps"] = [int(0.4*model_cfg["epochs"]), int(0.8*model_cfg["epochs"])]

    # WandB – Initialize a new run
    wandb.login()
    wandb.init(
        project=model_cfg["project"],
        group=model_cfg["model_arch"],
        name=model_cfg["name"],
    )
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    for k, v in model_cfg.items():
        setattr(config, k, v)
    return model_cfg, config

########################################################
#################### DATA UTILITIES ####################
########################################################


def prepareTorchDataset(model_cfg):
    """
    Prepare a torch dataset, parameterized by the model config

    Arguments:
    - model_cfg (wandb.Config): the wandb Config object

    Returns:
    - trainloader (torch.utils.data.DataLoader): training torch DataLoader
    - testloader (torch.utils.data.DataLoader): testing torch DataLoader
    """
    # Hacky fix for turning off SSL verification to handle URLError
    ssl._create_default_https_context = ssl._create_unverified_context

    # Define transforms
    transform_train_list = []
    transform_test_list = []

    # AutoAugmentation Policy for all tasks except generative ones
    if "gan" not in model_cfg["name"]:
        transform_train.append(transforms.AutoAugment(
                policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10, 
                interpolation=transforms.functional.InterpolationMode.BILINEAR
            )
        )
    
    # Ensure Tensors
    transform_train_list.append(transforms.ToTensor())
    transform_test_list.append(transforms.ToTensor())

    # Resize Tensors
    if not isinstance(model_cfg["imgsz"], list):
        model_cfg["imgsz"] = [model_cfg["imgsz"]]
    transform_train_list.append(transforms.Resize(model_cfg["imgsz"][1:], antialias=True))
    transform_test_list.append(transforms.Resize(model_cfg["imgsz"][1:], antialias=True))

    # Normalize Tensors
    # transform_train_list.append(
    #     transforms.Normalize(
    #         (0.5, 0.5, 0.5), 
    #         (0.5, 0.5, 0.5)
    #     )
    # )
    # transform_test_list.append(
    #     transforms.Normalize(
    #         (0.5, 0.5, 0.5), 
    #         (0.5, 0.5, 0.5)
    #     )
    # )

    ############ Temp for MNIST ############
    transform_train_list.append(
        transforms.Normalize((0.1307,), (0.3081,))
    )
    transform_test_list.append(
        transforms.Normalize((0.1307,), (0.3081,))
    )
    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)
    
    # Read the datasets for training and testing
    try:
        dataset_name_split = model_cfg["dataset_name"].split("/")
        dataset_source = str(dataset_name_split[0])
        dataset_name = str(dataset_name_split[1])
        data_root = f"./datasets/{model_cfg['task']}/{dataset_source}"
        trainset = getattr(torchvision.datasets, dataset_name)(
            root=data_root, 
            train=True,
            download=True, 
            transform=transform_train
        )
        testset = getattr(torchvision.datasets, dataset_name)(
            root=data_root, 
            train=False,
            download=True, 
            transform=transform_test
        )
    except Exception as e1:
        try:
            trainset = getattr(torchvision.datasets, dataset_name)(
                root=data_root, 
                split="train",
                download=True, 
                transform=transform_train
            )
            testset = getattr(torchvision.datasets, dataset_name)(
                root=data_root, 
                split="valid",
                download=True, 
                transform=transform_test
            )
        except Exception as e2:
            print("Error Loading Torch Dataset, dataset does not have argument train or split")
            print(e1)
            print(e2)
            sys.exit(1)

    # Create the corresponding dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=model_cfg.batch_size,
        shuffle=True, 
        pin_memory=True,
        num_workers=8
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=model_cfg.test_batch_size,
        shuffle=False, 
        pin_memory=True,
        num_workers=2
    )

    return trainloader, testloader


def prepareRoboFlowDataset(model_cfg):
    """
    Prepare a RoboFlow Dataset in the yolov8 annotation format

    Arguments:
    - model_cfg (wandb.Config): the wandb Config object
    """
    rf_key = os.getenv("ROBOFLOW_API_KEY")
    if rf_key is None:
        rf_key = input("Roboflow API Key: ")
        os.environ["ROBOFLOW_API_KEY"] = rf_key
    rf = Roboflow(api_key=rf_key)

    try:
        ws, pr, vs = model_cfg["dataset_name"].split("/")
        project = rf.workspace(ws).project(pr)
        dataset = project.version(vs).download(model_format="yolov8", location=f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/", overwrite=False)
    except:
        print("For roboflow datasets, you must specify the workspace, project, and version in the model_cfg.json field=dataset_name")
        sys.exit(1)


def prepareDetectionDataset(model_cfg):
    """
    Simple Function for ensuring that dataset exists and is saved locally

    Arguments:
    - model_cfg (wandb.Config): the wandb Config object
    """
    dataset_path = f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/"
    if not os.path.exists(dataset_path):
        prepareRoboFlowDataset(model_cfg)


def coco2yolo(model_cfg):
    """
    Function to convert MS-COCO Labels to YOLOv8 format
    """
    data_root = f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/"
    for root, folders, files in os.walk(data_root):
        if "coco" in root:
            if "train.json" in files:
                nc_list = [] # Collect list of classes
                yolo_dir = root.replace("coco", "yolo")
                os.makedirs(os.path.join(yolo_dir, "train", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "train", "labels"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "valid", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "valid", "labels"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "test", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "test", "labels"), exist_ok=True)

                ##### Go through train.json, val.json, test.json
                for dataset_json in ["train", "val", "test"]:
                    dataset_json_fpath = os.path.join(root, dataset_json)
                    with open(dataset_json_fpath + ".json", "r") as f:
                        dataset_json_dict = json.load(f)
                        id_sorted_ims = sorted(dataset_json_dict["images"], key=lambda im: im["id"])
                    if dataset_json == "val":
                        dataset_json = "valid"
                    ##### Go though the "images" and "annotations" keys
                    with tqdm(iterable=id_sorted_ims) as tq:
                        for im in id_sorted_ims:
                            coco_image_fpath = os.path.join(root, "images", im["file_name"])
                            yolo_image_fpath = os.path.join(yolo_dir, dataset_json, "images", im["file_name"])
                            coco_label_fpath = os.path.join(root, "annotations", im["file_name"].replace("jpg", "json"))
                            yolo_label_fpath = os.path.join(yolo_dir, dataset_json, "labels", im["file_name"].replace("jpg", "txt"))

                            ##### Copy image to corresponding images folder
                            try:
                                shutil.copy(coco_image_fpath, yolo_image_fpath)
                            except Exception as e:
                                print(e)
                                continue

                            ##### Create txt file of same name in a labels folder
                            with open(coco_label_fpath, "r") as coco_f:
                                coco_label_dict = json.load(coco_f)
                            
                            img_wh = [coco_label_dict["imageWidth"], coco_label_dict["imageHeight"]] # Use this to normalize labels
                            yolo_label_list = []
                            for shape in coco_label_dict["shapes"]: # for each instance
                                if shape["label"] not in nc_list:
                                    nc_list.append(shape["label"]) # add the class to our list if we haven't seen it before
                                label_ind = nc_list.index(shape["label"])
                                points = [str(label_ind)]
                                for pt in shape["points"]:
                                    points.extend([str(pt[0] / img_wh[0]), str(pt[1] / img_wh[1])])
                                yolo_label_list.append(" ".join(points))
                            with open(yolo_label_fpath, "w") as yolo_f:
                                yolo_f.write("\n".join(yolo_label_list))
                            tq.update()
                # Create data.yaml file
                dataset_yaml = {
                    "nc": len(nc_list),
                    "names": nc_list,
                    "path": yolo_dir.replace("./datasets/", "./"),
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images"
                }
                with open(os.path.join(yolo_dir, "data.yaml"), "w") as yolo_f:
                    yaml.dump(dataset_yaml, yolo_f)

###############################################################
#################### ALGORITHMIC UTILITIES ####################
###############################################################


def lr_finder_algo(net, optimizer, scheduler, criterion, train_loader, model_cfg):
	"""
    Function to perform a search for the optimal initial learning rate for the Cosine-Annealing with Warmup scheduler.
    Algorithm is implemented according to https://arxiv.org/abs/1506.01186
    
    Arguments:
    - model (torch.nn.Module): the torch model to be trained
    - optimizer (torch.optim): the torch optimizer to use
    - scheduler (torch.optim.lr_scheduler)
    - criterion (torch.nn.Loss): the torch loss function
    - train_loader (torch.utils.data.DataLoader): the torch Dataloader
    - model_cfg (wandb.Config): the wandb Config object
    
    Returns:
    - eta_max (float): the maximized initial learning rate
	"""
	device, epochs = model_cfg["device"], max(150, model_cfg["epochs"])
	model = net.to(device)
	losses = []
	lr_list = []
	for epoch, (images, labels) in enumerate(train_loader):
		if epoch == epochs:
			break
		# Move tensors to configured device
		images = images.to(device)
		labels = labels.to(device)
		# Forward Pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		# Backpropogation and SGD
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		curr_lr = optimizer.param_groups[0]['lr']
		wandb.log(
			{
				"Training Loss (LR Finder)": loss.item(),
				"LR": curr_lr,
				"Epoch": epoch
			}
		)
		losses.append(loss.item())
		lr_list.append(curr_lr)
		scheduler.step()
	eta_max = lr_list[min(enumerate(losses), key=lambda x: x[1])[0]] / 10.0
	return eta_max


def custom_cos_annealing_warmup(num_batches, T, eta_max, epoch, batch):
    """
    Custom Implementation of a cosine annealing learning rate scheduler with warmup
    
    Arguments:
    - num_batches (int): number of batches in the dataset
    - T (int): num_batches*epochs
    - eta_max (float): the maximized initial learning rate
    - epoch (int): the current epoch
    - batch (int): the current batch number/id
    
    Returns:
    - next_lr (float): the next scheduled learning rate
	"""
	# fixed hyperparameters used for Cosine Annealing w/ Warmup Schedule
    T_0 = int(T / 5)
    t = epoch*num_batches + batch
    next_lr = 1e-5
    if t <= T_0:
        next_lr += (t / T_0)*eta_max
    else:
        next_lr += eta_max*math.cos((math.pi/2.0)*((t - T_0)/(T - T_0)))
    return next_lr