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

from classification.models import custom_models
from classification.models.custom_models.FCN.fcn import FCN


#########################################################
#################### TRAINING ENGINE ####################
#########################################################

if __name__ == "__main__":

	cfg_fname = "./classification/model_cfg.json"
	cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

	# Set random seeds for reproducability
	torch.manual_seed(cfg_wandb.seed)
	random.seed(cfg_wandb.seed)
	np.random.seed(cfg_wandb.seed)

	trainloader, testloader = prepareTorchDataset(cfg_wandb)

	# Initialize model
	model = FCN().to(cfg_dict["device"])
	# Define loss function
	criterion = getattr(nn, cfg_wandb.criterion)()
	# Set optimizer
	optimizer = getattr(
		optim, 
		cfg_wandb.optimizer
		)(
			model.parameters(), 
			lr=cfg_wandb.lr, 
			momentum=cfg_wandb.momentum, 
			weight_decay=cfg_wandb.weight_decay, 
			nesterov=cfg_wandb.nesterov
	) 
	# Set scheduler
	scheduler = getattr(
		optim.lr_scheduler,
		cfg_wandb.scheduler
		)(
			optimizer=optimizer, 
			milestones=cfg_wandb.lr_steps
	)

	# Training loop called here
	train(
		model, 
		optimizer, 
		scheduler, 
		criterion, 
		trainloader, 
		testloader, 
		cfg_dict
	)
	# Save the model
	final_save_name = f"./classification/models/custom_models/{cfg_dict['wandb']['group']}/{cfg_dict['wandb']['name']}_final.h5"
	torch.save(model.state_dict(), final_save_name)

