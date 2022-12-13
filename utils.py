#################################################
#################### IMPORTS ####################
#################################################

import math
import json
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import wandb


###########################################################
#################### UTILITY FUNCTIONS ####################
###########################################################

# Initialize Weights and Biases for Experiment Tracking
def initialize_wandb(cfg_fname):
	wandb.login()
	with open(cfg_fname, "r") as f:
		model_cfg = json.load(f)

	# WandB – Initialize a new run
	wandb.init(project=model_cfg["wandb"]["project"],
			group=model_cfg["wandb"]["group"],
			name=model_cfg["wandb"]["name"])
	wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

	# WandB – Config is a variable that holds and saves hyperparameters and inputs
	config = wandb.config
	for k, v in model_cfg["hyperparameters"].items():
		setattr(config, k, v)
	config.model_name = model_cfg["wandb"]["group"] # Save folder for model checkpoints
	config.lr_steps = [int(0.4*config.epochs), int(0.8*config.epochs)]
	model_cfg["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	return model_cfg, config


# Cosine Annealing w/ Warmup Learning Rate Schedule
def lr_scheduler_impl(num_batches, epoch, i):
	t = epoch*num_batches + i
	next_lr = 1e-5
	if t <= T_0:
		next_lr += (t / T_0)*eta_max
	else:
		next_lr += eta_max*math.cos((math.pi/2.0)*((t - T_0)/(T - T_0)))
	return next_lr


def prepareTorchDataset(model_cfg):
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