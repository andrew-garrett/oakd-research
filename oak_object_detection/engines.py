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

from utils import prepareTorchDataset, lr_finder_algo, initialize_wandb, prepareDetectionDataset
# from oak_classification.train import *
# from oak_classification.test import *

import oak_object_detection.models.custom_models as custom_models


#########################################################
#################### TRAINING ENGINE ####################
#########################################################

def base_engine(cfg_fname):
	
    cfg_dict, cfg_wandb = initialize_wandb(cfg_fname)

    prepareDetectionDataset(cfg_wandb)

    # Set random seeds for reproducability
    torch.manual_seed(cfg_wandb.seed)
    random.seed(cfg_wandb.seed)
    np.random.seed(cfg_wandb.seed)

    # Initialize model
    model_arch = cfg_dict["model_arch"].upper()
    try:
        model = getattr(
            getattr(
                getattr(
                    custom_models, 
                    model_arch,
                ),
                model_arch
            ), model_arch
        )(cfg_dict)
    except Exception as e:
        print(e)
        return
	
    model.train()