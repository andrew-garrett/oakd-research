#################################################
#################### IMPORTS ####################
#################################################

import lightning.pytorch as pl
import torch
import torch.nn as nn

############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################

class VAE(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)