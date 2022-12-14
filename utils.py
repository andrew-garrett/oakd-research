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

	# WandB – Initialize a new run
	wandb.init(project=f"oakd-research-{model_cfg['task'].replace('_', '-')}",
			group=model_cfg["model_arch"],
			name=model_cfg["name"])
	wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

	# WandB – Config is a variable that holds and saves hyperparameters and inputs
	config = wandb.config
	for k, v in model_cfg.items():
		setattr(config, k, v)
	config.lr_steps = [int(0.4*config.epochs), int(0.8*config.epochs)]
	model_cfg["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# num_batches = len(trainloader)
# T = config.epochs * num_batches
# T_0 = int(T / 5)

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