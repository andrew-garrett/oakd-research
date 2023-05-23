#################################################
#################### IMPORTS ####################
#################################################

import glob
import math
import json
import os
import ssl
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
from roboflow import Roboflow


###########################################################
#################### UTILITY FUNCTIONS ####################
###########################################################

# Initialize Weights and Biases for Experiment Tracking
def initialize_wandb(cfg_fname):
	"""
	Function to initialize wandb experiment, configured by the specified config file

	Arguments:
		- cfg_fname (string): specifies the file location of model_cfg.json

	Returns:
		- model_cfg (dict): dictionary storing data from model_cfg.json
		- config (wandb.Config): config object for wandb experiment
	"""
	wandb.login()
	with open(cfg_fname, "r") as f:
		model_cfg = json.load(f)
	model_cfg["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# WandB – Initialize a new run
	wandb.init(
		project=f"oakd-research-{model_cfg['task'].replace('_', '-')}",
		group=model_cfg["model_arch"],
		name=model_cfg["name"],
	)
	# WandB – Config is a variable that holds and saves hyperparameters and inputs
	config = wandb.config
	for k, v in model_cfg.items():
		setattr(config, k, v)
	config.lr_steps = [int(0.4*config.epochs), int(0.8*config.epochs)]
	return model_cfg, config


def lr_finder_algo(net, optimizer, scheduler, criterion, train_loader, model_cfg):
	"""
	Function to perform a search for the optimal initial learning rate for the
	Cosine-Annealing w/ Warmup scheduler.
	"""
	device, epochs = model_cfg["device"], max(150, model_cfg["epochs"])
	model = net.to(device)
	losses = []
	lr_list = []
	for epoch, (images, labels) in enumerate(train_loader):
		if epoch == epochs:
			break;
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
	eta_max = lr_list[min(enumerate(losses), key=lambda x: x[1])[0]] / 10.0 # losses.index(min(losses))
	return eta_max


# Cosine Annealing w/ Warmup Learning Rate Schedule
def lr_scheduler_impl(num_batches, T, eta_max, epoch, i):
	# fixed hyperparameters used for Cosine Annealing w/ Warmup Schedule
	T_0 = int(T / 5)
	t = epoch*num_batches + i
	next_lr = 1e-5
	if t <= T_0:
		next_lr += (t / T_0)*eta_max
	else:
		next_lr += eta_max*math.cos((math.pi/2.0)*((t - T_0)/(T - T_0)))
	return next_lr


def prepareTorchDataset(model_cfg):
	# Hacky fix for turning off SSL verification to handle URLError
	ssl._create_default_https_context = ssl._create_unverified_context

	#  Define classes in the CIFAR dataset
	classes = (
		'plane', 'car', 'bird', 'cat', 'deer', 
		'dog', 'frog', 'horse', 'ship', 'truck'
	)

	# Define transforms, Read the datasets for training and testing, and Create the corresponding dataloaders
	transform_train = transforms.Compose(
		[
			transforms.AutoAugment(
				policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10, 
				interpolation=transforms.functional.InterpolationMode.BILINEAR
			),
			transforms.ToTensor(),
			transforms.Normalize(
				(0.5, 0.5, 0.5), 
				(0.5, 0.5, 0.5)
			)
		]
	)
	trainset = torchvision.datasets.CIFAR10(
		root='./data', 
		train=True,
		download=True, 
		transform=transform_train
	)
	trainloader = torch.utils.data.DataLoader(
		trainset, 
		batch_size=model_cfg.batch_size,
		shuffle=True, 
		pin_memory=True
	)

	# Define transforms, Read the datasets for training and testing, and Create the corresponding dataloaders
	transform_test = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(
				(0.5, 0.5, 0.5), 
				(0.5, 0.5, 0.5)
			)
		]
	)
	testset = torchvision.datasets.CIFAR10(
		root='./data', 
		train=False,
		download=True, 
		transform=transform_test
	)
	testloader = torch.utils.data.DataLoader(
		testset, 
		batch_size=model_cfg.test_batch_size,
		shuffle=False, 
		pin_memory=True
	)

	return trainloader, testloader

def prepareRoboFlowDataset(model_cfg):

	rf_key = os.getenv("ROBOFLOW_API_KEY")
	if rf_key is None:
		rf_key = input("Roboflow API Key: ")
		os.environ["ROBOFLOW_API_KEY"] = rf_key
	rf = Roboflow(api_key=rf_key)

	ws, pr, vs = model_cfg["dataset_name"].split("/")
	project = rf.workspace(ws).project(pr)
	dataset = project.version(vs).download(model_format="yolov8", location=f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/", overwrite=False)


def prepareDetectionDataset(model_cfg):

	dataset_path = f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/"
	if not os.path.exists(dataset_path):
		try:
			prepareRoboFlowDataset(model_cfg)
		except Exception as e:
			print(e)
			return


# Convert between coco json and yolo format
def coco2json(model_cfg):
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