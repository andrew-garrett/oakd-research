#################################################
#################### IMPORTS ####################
#################################################

import json
import os
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb

from utils import prepareTorchDataset, lr_finder_algo
from oak_classification.train import *
from oak_classification.test import *


import oak_classification.models.custom_models as custom_models


#########################################################
#################### TRAINING ENGINE ####################
#########################################################

def base_engine(cfg_fname):

	cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

	# Set random seeds for reproducability
	torch.manual_seed(cfg_wandb.seed)
	random.seed(cfg_wandb.seed)
	np.random.seed(cfg_wandb.seed)

	trainloader, testloader = prepareTorchDataset(cfg_wandb)

	# Initialize model
	model_arch = cfg_dict["model_arch"]
	try:
		model = getattr(
			getattr(
				getattr(
					custom_models, 
					model_arch,
				),
				model_arch
			), model_arch
		)().to(cfg_dict["device"])
	except Exception as e:
		print(e)
		return
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
	try:
		# Set scheduler
		scheduler = getattr(
			optim.lr_scheduler,
			cfg_wandb.scheduler
			)(
				optimizer=optimizer, 
				milestones=cfg_wandb.lr_steps
		)
	except:
		scheduler = None
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
	final_save_name = f"./oak_{cfg_dict['task']}/models/custom_models/{cfg_dict['model_arch']}/{cfg_dict['name']}_final.h5"
	torch.save(model.state_dict(), final_save_name)



def lr_finding_engine(cfg_fname):
	with open(cfg_fname, "r") as f:
		tmp_cfg_dict = json.load(f)
		tmp_cfg_dict["model_arch"] += "_LRF"
	with open(cfg_fname, "w") as f:
		json.dump(tmp_cfg_dict, f, indent="\t")

	cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

	# Set random seeds for reproducability
	torch.manual_seed(cfg_wandb.seed)
	random.seed(cfg_wandb.seed)
	np.random.seed(cfg_wandb.seed)

	# Load data
	trainloader, testloader = prepareTorchDataset(cfg_wandb)

	# Initialize model
	model_arch = cfg_dict["model_arch"].upper()
	model_arch = model_arch.replace("_LRF", "")
	try:
		model = getattr(
			getattr(
				getattr(
					custom_models, 
					model_arch,
				),
				model_arch
			), model_arch
		)().to(cfg_dict["device"])
	except Exception as e:
		print(e)
		return
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
	lr_fcn = lambda epoch: 1.1
	scheduler = optim.lr_scheduler.MultiplicativeLR(
		optimizer=optimizer, 
		lr_lambda=lr_fcn, 
		last_epoch=-1
	)

	# Find the optimal learning rate
	eta_max = lr_finder_algo(
		model, 
		optimizer, 
		scheduler, 
		criterion, 
		trainloader, 
		cfg_dict
	)
	
	# Write it to the config
	with open(cfg_fname, "r") as f:
		tmp_cfg_dict = json.load(f)
		tmp_cfg_dict["model_arch"] = tmp_cfg_dict["model_arch"][:-len("_LRF")]
		tmp_cfg_dict["lr"] = eta_max
	with open(cfg_fname, "w") as f:
		json.dump(tmp_cfg_dict, f, indent="\t")

	base_engine(cfg_fname)


if __name__ == "__main__":

	cfg_fname = "./oak_classification/model_cfg.json"
	# engine(cfg_fname)
	# lr_finding_engine(cfg_fname)
