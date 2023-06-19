#################################################
#################### IMPORTS ####################
#################################################


import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from math import log2

import wandb

from oak_generation.models.custom_models.DCGAN.DCGAN import DCDiscriminator, DCGenerator
from oak_generation.models.custom_models.WGAN.WGAN import WGAN

############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################

class ConditionalDCDiscriminator(DCDiscriminator):
    """
    Pytorch nn.Module Conditional Discriminator Subnetwork for a DCGAN
    """
    
    def __init__(self, img_shape, num_classes):
        # Treat the Conditioning as an additional channel for model inputs
        img_shape = list(img_shape)
        img_shape[0] += 1
        super().__init__(img_shape)
        # Use number of classes to condition the GAN
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, self.img_shape[1]*self.img_shape[2])

    def forward(self, img, label):
        # Use embedding as an extra channel of the image
        embedding = self.embed(label).view(len(label), 1, self.img_shape[1], self.img_shape[2])
        x = torch.cat([img, embedding], dim=1) # N x (C + 1) x H x W
        return self.model(x) 


class ConditionalDCGenerator(DCGenerator):
    """
    Pytorch nn.Module Conditional Generator Subnetwork for a DCGAN
    """

    def __init__(self, latent_dim, img_shape, num_classes):
        # The embedding is added to the latent vector as additional "channels"
        super().__init__(2*latent_dim, img_shape)
        self.num_classes = num_classes
        # Unlike the Discriminator, the embedding will be outputted by the model
        self.embed = nn.Embedding(num_classes, latent_dim)

    def forward(self, z, label):
        # Use embedding as an extra channel of the latent vector
        embedding = self.embed(label).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, embedding], dim=1) # N x (LATENT_DIM + EMBEDDING_DIM) x H x W
        return self.model(x)


class CGAN(WGAN):
    """
    Conditional WGAN with Gradient Penalty
    """
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        # Subnetworks
        self.generator = ConditionalDCGenerator(
            latent_dim=self.hparams.model_cfg["latent_dim"], 
            img_shape=tuple(self.hparams.model_cfg["imgsz"]),
            num_classes=self.hparams.model_cfg["num_classes"]
        )
        self.discriminator = ConditionalDCDiscriminator(
            img_shape=tuple(self.hparams.model_cfg["imgsz"]),
            num_classes=self.hparams.model_cfg["num_classes"]
        )

        # Weight Initialization According to DCGAN Paper
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Example Input
        self.example_input_array = (
            torch.zeros(2, self.hparams.model_cfg["latent_dim"], 1, 1),
            torch.ones(2, dtype=torch.int)
        )

    def _generator_loss(self, z, labels):
        """
        WGAN with Gradient Penalty Loss Function (Conditional)
        """
        fake_imgs = self(z, labels)
        return -1. * torch.mean(self.discriminator(fake_imgs, labels))

    def _discriminator_loss(self, imgs, labels, z):
        """
        WGAN with Gradient Penalty Loss Function (Conditional)
        """
        d_real = self.discriminator(imgs, labels).view(-1)
        fake_imgs = self(z, labels).detach()
        d_fake = self.discriminator(fake_imgs, labels).view(-1)
        return torch.mean(d_fake - d_real + self.lambda_gp * self._gradient_penalty(imgs, fake_imgs, labels))

    def _generator_training_step(self, labels, optimizer_g, z):
        """
        Generator Weight Update Step (Conditional)
        """
        # generate images
        self.toggle_optimizer(optimizer_g)
        # save the images without the embedding layer
        self.generated_imgs = self(z, labels).detach()
        # compute loss, perform backward pass, and step optimizer
        g_loss = self._generator_loss(z, labels)
        self.manual_backward(g_loss)
        # self.on_before_optimizer_step(optimizer_g, mode="generator")
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        return g_loss

    def _discriminator_training_step(self, imgs, labels, optimizer_d, z):
        """
        Discriminator Weight Update Step (Conditional)
        """
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)
        # compute loss, perform backward pass, and step optimizer
        d_loss = self._discriminator_loss(imgs, labels, z)
        self.manual_backward(d_loss)
        # self.on_before_optimizer_step(optimizer_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        return d_loss

    def forward(self, z, label):
        return self.generator(z, label)
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        optimizer_g, optimizer_d = self.optimizers()

        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.model_cfg["latent_dim"], 1, 1)
        z = z.type_as(imgs)

        # Train Generator
        g_loss = self._generator_training_step(labels, optimizer_g, z)

        # Train Discriminator
        d_loss = 0.
        for _ in range(self.discriminator_updates):
            d_loss += self._discriminator_training_step(imgs, labels, optimizer_d, z)
        d_loss /= self.discriminator_updates

        # Real/Fake Image Logging
        if len(self.sample_imgs) < 6:
            # log sampled images
            sample_inds = torch.randint(self.generated_imgs.shape[0], (6,))
            sample_imgs = self.generated_imgs[sample_inds]
            self.sample_imgs.append(sample_imgs)
            # log training images
            train_imgs = imgs[sample_inds]
            self.train_imgs.append(train_imgs)
        else:
            if batch_idx % 100 == 0:
                self._log_images()
        
        # Loss Logging
        self.log_dict(
            {
                "train/g_loss": g_loss,
                "train/d_loss": d_loss
            }, 
            prog_bar=True
        )
        self.loss_dict["train/g_loss_epoch"].append(g_loss)
        self.loss_dict["train/d_loss_epoch"].append(d_loss)