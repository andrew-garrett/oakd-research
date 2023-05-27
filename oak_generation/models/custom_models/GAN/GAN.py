#################################################
#################### IMPORTS ####################
#################################################


import lightning.pytorch as pl
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

import wandb

############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################


class Discriminator(nn.Module):
    """
    Pytorch nn.Module Discriminator Subnetwork for a Vanilla GAN
    """
    
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Generator(nn.Module):
    """
    Pytorch nn.Module Generator Subnetwork for a Vanilla GAN
    """

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def generator_block(in_feat, out_feat, normalize=True):
            block = [nn.Linear(in_feat, out_feat)]
            if normalize:
                block.append(nn.BatchNorm1d(out_feat)) # 0.8
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
            *generator_block(latent_dim, 128, normalize=False),
            *generator_block(128, 256),
            *generator_block(256, 512),
            *generator_block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class GAN(pl.LightningModule):
    """
    Pytorch Lightning Vanilla GAN, based on tutorial at https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/basic-gan.html
    """

    def __init__(self, 
                 model_cfg, 
                 latent_dim: int = 100,
                 **kwargs,
    ):
        super(GAN, self).__init__(**kwargs)

        # Log arguments
        self.save_hyperparameters()

        # Use this to perform the optimization in the training step
        self.automatic_optimization = False

        # Subnetworks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=tuple(self.hparams.model_cfg["imgsz"]))
        self.discriminator = Discriminator(img_shape=tuple(self.hparams.model_cfg["imgsz"]))

        # Setup data structures for 
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        self.tensor2PIL = ToPILImage()

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        if len(self.sample_imgs) < 6:
            sample_imgs = self.generated_imgs[:6]
            self.sample_imgs.append(sample_imgs)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Logging
        self.log_dict(
            {
                "train/g_loss": g_loss,
                "train/d_loss": d_loss
            }, 
            prog_bar=True
        )
        self.loss_dict["train/g_loss_epoch"].append(g_loss)
        self.loss_dict["train/d_loss_epoch"].append(d_loss)


    def configure_optimizers(self):
        """
        optimizer = optimizers[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        opts = {"optimizer": optimizer}
        if self.use_scheduler:
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=3, threshold=1e-4, threshold_mode="rel"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss_epoch"
            }
            opts["lr_scheduler"] = scheduler_config
        return opts
        """

        optimizer = self.hparams.model_cfg["optimizer"]
        lr = self.hparams.model_cfg["lr"]
        weight_decay = self.hparams.model_cfg["weight_decay"]

        if optimizer in ("SGD", "Adadelta"):
            betas = self.hparams.model_cfg["momentum"]
        elif optimizer in ("Adam", "AdamW", "Adamax", ):
            betas = (self.hparams.model_cfg["momentum"], 0.999)

        opt_g = getattr(torch.optim, self.hparams.model_cfg["optimizer"])(self.generator.parameters(), lr, betas, weight_decay=weight_decay)
        opt_d = getattr(torch.optim, self.hparams.model_cfg["optimizer"])(self.discriminator.parameters(), lr, betas, weight_decay=weight_decay)
        opts = [opt_g, opt_d]

        if self.hparams.model_cfg["scheduler"]:
            scheduler_config_g = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.25, patience=3, threshold=1e-4, threshold_mode="rel"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/g_loss",
                "name": "G Scheduler",
            }
            scheduler_config_d = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.25, patience=3, threshold=1e-4, threshold_mode="rel"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/d_loss",
                "name": "D Scheduler",
            }

            schedulers = [scheduler_config_g, scheduler_config_d]
            opts = (opts, schedulers)

        return opts
    
    def on_train_epoch_start(self):

        self.loss_dict = {
            "train/g_loss_epoch": [],
            "train/d_loss_epoch": [],
        }
        self.sample_imgs = []

    def on_train_epoch_end(self):

        self.loss_dict["train/g_loss_epoch"] = torch.mean(torch.tensor(self.loss_dict["train/g_loss_epoch"]))
        self.loss_dict["train/d_loss_epoch"] = torch.mean(torch.tensor(self.loss_dict["train/d_loss_epoch"]))
        self.log_dict(
            dictionary=self.loss_dict,
            on_epoch=True,
        )
        grid = make_grid(torch.vstack(self.sample_imgs), nrow=6)
        wandb.log(
            {"generated_images": wandb.Image(self.tensor2PIL(grid))}
        )
        

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)