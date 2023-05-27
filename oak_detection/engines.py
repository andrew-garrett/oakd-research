#################################################
#################### IMPORTS ####################
#################################################

import random

import numpy as np
import torch

import oak_detection.models.custom_models as custom_models
from utils import initialize_wandb, prepareDetectionDataset

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
            custom_models, 
            model_arch,
        )(cfg_dict)
    except Exception as e:
        print(e)
        return

    model.train()