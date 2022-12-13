#################################################
#################### IMPORTS ####################
#################################################

import json
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb

from utils import prepareTorchDataset
from classification.train import *
from classification.test import *

from models import custom_models


#########################################################
#################### TRAINING ENGINE ####################
#########################################################

if __name__ == "__main__":

	cfg_fname = "./model_cfg.json"
	cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

	# Set random seeds for reproducability
	torch.manual_seed(cfg_wandb.seed)
	random.seed(cfg_wandb.seed)
	np.random.seed(cfg_wandb.seed)

	trainloader, testloader = prepareTorchDataset(cfg_wandb)

	# Initialize a new model
	model = allcnn_t().to(device)
	# Define the loss function as asked in the question
	criterion = nn.CrossEntropyLoss()
	criterion = getattr(nn, cfg_wandb.criterion + "()")
	# Set optimizer and scheduler
	# optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov)
	optimizer = getattr(
		optim, 
		f"{cfg_wandb.optimizer}(\
			{model.parameters()}, \
			lr={cfg_wandb.lr}, \
			momentum={cfg_wandb.momentum}, \
			weight_decay={cfg_wandb.weight_decay}, \
			nesterov={cfg_wandb.nesterov})"
	)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.lr_steps)
	scheduler = getattr(
		optim.lr_scheduler,
		f"{cfg_wandb.scheduler}(\
			optimizer={optimizer}, \
			milestones={cfg_wandb.lr_steps}"
	)
	# Training loop called here
	train(
		model, 
		optimizer, 
		scheduler, 
		criterion, 
		trainloader, 
		testloader, 
		config.epochs, 
		config.model_name)
	# Save the model
	final_model_name = root_dir + 'models/' + config.model_name + '/final.h5'
	torch.save(model.state_dict(), final_model_name)