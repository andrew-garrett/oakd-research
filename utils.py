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

	ws, pr = model_cfg["dataset_name"].split("/")
	project = rf.workspace(ws).project(pr)
	dataset = project.version(1).download(model_format="yolov8", location=f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/", overwrite=False)


def prepareDetectionDataset(model_cfg):

	dataset_path = f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/"
	if not os.path.exists(dataset_path):
		try:
			prepareRoboFlowDataset(model_cfg)
		except Exception as e:
			print(e)
			return
	
	# We've now ensured we have a data.yaml file, let's ensure that the image size is accounted for
	# import yaml
	# with open(os.path.join(dataset_path, "data.yaml"), "r") as f:
	# 	dataset_yaml = yaml.safe_load(f)
	# 	if str(model_cfg['imgsz']) not in dataset_yaml["train"]:
	# 		# training data
	# 		train_dir = dataset_yaml["train"].replace("train", f"train_{model_cfg['imgsz']}").replace("/images", "")
	# 		dataset_yaml["train"] = train_dir + "/images"
	# 		train_dir = os.path.join(dataset_path, train_dir)
	# 	else:
	# 		train_dir = os.path.join(dataset_path, dataset_yaml["train"].replace("/images", ""))
	# 	if str(model_cfg['imgsz']) not in dataset_yaml["val"]:
	# 		# validation data
	# 		val_dir = dataset_yaml["val"].replace("valid", f"valid_{model_cfg['imgsz']}").replace("/images", "")
	# 		dataset_yaml["val"] = val_dir + "/images"
	# 		val_dir = os.path.join(dataset_path, val_dir)
	# 	else:
	# 		val_dir = os.path.join(dataset_path, dataset_yaml["val"].replace("/images", ""))
	# 	if str(model_cfg['imgsz']) not in dataset_yaml["test"]:
	# 		# testing data
	# 		test_dir = dataset_yaml["test"].replace("test", f"test_{model_cfg['imgsz']}").replace("/images", "")
	# 		dataset_yaml["test"] = test_dir + "/images"
	# 		test_dir = os.path.join(dataset_path, test_dir)
	# 	else:
	# 		test_dir = os.path.join(dataset_path, dataset_yaml["test"].replace("/images", ""))

	# with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
	# 	yaml.dump(dataset_yaml, f)
	
	# try:
	# 	os.mkdir(train_dir)
	# 	os.mkdir(val_dir)
	# 	os.mkdir(test_dir)
	# 	# Resize images in the dataset
	# 	num_files = glob.glob(dataset_path + "train/**/*.png") + glob.glob(dataset_path + "valid/**/*.png") + glob.glob(dataset_path + "test/**/*.png")
	# 	num_files += glob.glob(dataset_path + "train/**/*.jpg") + glob.glob(dataset_path + "valid/**/*.jpg") + glob.glob(dataset_path + "test/**/*.jpg")
	# 	with tqdm(iterable=num_files) as tq:
	# 		for filename in num_files:
	# 			try:
	# 				cv2_im = cv2.imread(filename=filename)
	# 				cv2_im = cv2.resize(cv2_im, (model_cfg['imgsz'], model_cfg['imgsz']), interpolation=cv2.INTER_AREA)
	# 				if "train" in filename:
	# 					new_savename = filename.replace("train", f"train_{model_cfg['imgsz']}")
	# 				elif "val" in filename:
	# 					new_savename = filename.replace("valid", f"valid_{model_cfg['imgsz']}")
	# 				elif "test" in filename:
	# 					new_savename = filename.replace("test", f"test_{model_cfg['imgsz']}")
	# 				else:
	# 					continue
	# 				os.makedirs(os.path.dirname(new_savename), exist_ok=True)
	# 				cv2.imwrite(filename=new_savename, img=cv2_im)
	# 				tq.update()
	# 			except:
	# 				continue
	# except:
	# 	print("Resized dataset already exists, no preprocessing necessary")
	# 	return
