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

############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################


class DCDiscriminator(nn.Module):
    """
    Pytorch nn.Module Discriminator Subnetwork for a DCGAN
    """
    
    def __init__(self, img_shape):
        super(DCDiscriminator, self).__init__()

        self.img_shape = img_shape

        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            ]
            if normalize:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        # Number of channels in the training images
        nc = img_shape[0]
        # Size of feature maps
        ndf = img_shape[1]

        # architecture is variable in the number of
        n_conv_layers = int(log2(ndf))
        # ensure the image dim is a power of 2
        assert (ndf and (not(ndf & (ndf - 1))) )

        # TODO: handle image sizes that are powers of 2
        # layers = [*discriminator_block(nc, ndf, normalize=False)]

        self.model = nn.Sequential(
            *discriminator_block(nc, ndf, normalize=False),
            *discriminator_block(ndf, 2*ndf),
            *discriminator_block(2*ndf, 4*ndf),
            *discriminator_block(4*ndf, 8*ndf),
            nn.Conv2d(8*ndf, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid(), # This is used in the DCGAN, but not in the WGAN or WGAN-GP
        )

    def forward(self, img):
        return self.model(img)


class DCGenerator(nn.Module):
    """
    Pytorch nn.Module Generator Subnetwork for a DCGAN
    """

    def __init__(self, latent_dim, img_shape):
        super(DCGenerator, self).__init__()

        self.img_shape = img_shape

        def generator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            block = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            return block
        
        # Number of channels in the training images
        nc = img_shape[0]
        # Size of feature maps
        ngf = img_shape[1]

        # architecture is variable in the number of
        n_conv_layers = int(log2(ngf))
        # ensure the image dim is a power of 2
        assert (ngf and (not(ngf & (ngf - 1))) )

        self.model = nn.Sequential(
            *generator_block(latent_dim, 8*ngf, stride=1, padding=0),
            *generator_block(8*ngf, 4*ngf),
            *generator_block(4*ngf, 2*ngf),
            *generator_block(2*ngf, ngf),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class DCGAN(pl.LightningModule):
    """
    Pytorch Lightning Vanilla GAN, based on tutorial at https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/basic-gan.html
    """

    def __init__(self, 
                 model_cfg,
                 **kwargs,
    ):
        super(DCGAN, self).__init__(**kwargs)

        # Log arguments
        self.save_hyperparameters()

        # Use this to perform the optimization in the training step
        self.automatic_optimization = False

        # Subnetworks
        self.generator = DCGenerator(latent_dim=self.hparams.model_cfg["latent_dim"], img_shape=tuple(self.hparams.model_cfg["imgsz"]))
        self.discriminator = DCDiscriminator(img_shape=tuple(self.hparams.model_cfg["imgsz"]))
        
        # Weight Initialization According to DCGAN Paper
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Setup data structures for 
        self.validation_z = torch.randn(8, self.hparams.model_cfg["latent_dim"])
        self.example_input_array = torch.zeros(2, self.hparams.model_cfg["latent_dim"], 1, 1)

        # Used to Log Images
        self.tensor2PIL = ToPILImage()

        # Discriminator Updates per Iteration (generally used for other architectures with deeper discriminators)
        self.discriminator_updates = 1

    # def on_before_optimizer_step(self, optimizer, mode="discriminator") -> None:
    #     norms = grad_norm(getattr(self, mode), norm_type=2, group_separator=f"/{mode}/")
    #     self.log_dict(norms)

    def forward(self, z):
        return self.generator(z)
    
    def _adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _generator_loss(self, z, valid):
        """
        DCGAN Generator Loss (BCE Loss of Fake Samples Labeled as Real)
        """
        return self._adversarial_loss(self.discriminator(self(z)).view(-1), valid)
    
    def _discriminator_loss(self, imgs, valid, z, fake):
        """
        DCGAN Discriminator Loss (Average BCE Loss of Real and Fake Samples)
        """
        real_loss = self._adversarial_loss(self.discriminator(imgs).view(-1), valid)
        fake_imgs = self(z)
        fake_loss = self._adversarial_loss(self.discriminator(fake_imgs.detach()).view(-1), fake)
        # discriminator loss is the average of these
        return (real_loss + fake_loss) / 2

    def _generator_training_step(self, imgs, optimizer_g, z):
        """
        Generator Weight Update Step
        """
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z).detach()

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0))
        valid = valid.type_as(imgs)

        g_loss = self._generator_loss(z, valid)
        self.manual_backward(g_loss)
        # self.on_before_optimizer_step(optimizer_g, mode="generator")
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        return g_loss

    def _discriminator_training_step(self, imgs, optimizer_d, z):
        """
        Discriminator Weight Update Step
        """
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0))
        valid = valid.type_as(imgs)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0))
        fake = fake.type_as(imgs)

        d_loss = self._discriminator_loss(imgs, valid, z, fake)
        self.manual_backward(d_loss)
        # self.on_before_optimizer_step(optimizer_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        return d_loss

    def _log_images(self):
        sample_grid = make_grid(torch.vstack(self.sample_imgs), nrow=6, normalize=True)
        train_grid = make_grid(torch.vstack(self.train_imgs), nrow=6, normalize=True)
        wandb.log(
            {
                "generated_images": wandb.Image(self.tensor2PIL(sample_grid)),
                "training_images": wandb.Image(self.tensor2PIL(train_grid))
            }
        )

        self.sample_imgs = []
        self.train_imgs = []


    def training_step(self, batch, batch_idx):

        # imgs, _ = batch
        # optimizer_g, optimizer_d = self.optimizers()

        # ##### Step 1. Update Discriminator Network
        # self.toggle_optimizer(optimizer_d)
        # optimizer_d.zero_grad()

        # ## Train with Real Labels
        # real_label = torch.ones(imgs.size(0))
        # real_label = real_label.type_as(imgs)
        # # Forward Pass
        # output_d = self.discriminator(imgs).view(-1)
        # # Compute Loss and Backward Pass
        # error_d_real = self._adversarial_loss(output_d, real_label)
        # # self.manual_backward(error_d_real)
        # d_x = output_d.mean().item()

        # ## Train with Fake Labels
        # z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1)
        # z = z.type_as(imgs)
        # # Create Fake Images with Generator
        # fake_imgs = self(z)
        # fake_label = torch.zeros(imgs.size(0))
        # fake_label = fake_label.type_as(imgs)
        # # Forward Passs
        # output_d = self.discriminator(self(z).detach()).view(-1)
        # # Compute Loss and Backward Pass
        # error_d_fake = self._adversarial_loss(output_d, fake_label)
        # # d_g_z1 = output_d.mean().item()

        # ## Perform Weight Update
        # error_d = error_d_real + error_d_fake
        # self.manual_backward(error_d)
        # optimizer_d.step()
        # self.untoggle_optimizer(optimizer_d)


        # ##### Step 2. Update Generator Network
        # self.toggle_optimizer(optimizer_g)
        # optimizer_g.zero_grad()
        # z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1)
        # z = z.type_as(imgs)
        # real_label = torch.ones(imgs.size(0))
        # real_label = real_label.type_as(imgs)

        # # Forward Pass (Classify Generated Images)
        # output_d2 = self.discriminator(self(z).detach()).view(-1)
        # # Compute Loss (Fake Images are "real" for Generator Cost) and Backward Pass
        # error_g = self._adversarial_loss(output_d2, real_label)
        # self.manual_backward(error_g)
        # # d_g_z2 = output_d2.mean().item()
        # optimizer_g.step()
        # self.untoggle_optimizer(optimizer_g)


        # ##### Step 3. Logging

        # # Generated and Training Images
        # if len(self.sample_imgs) < 6:
        #     # log sampled images
        #     sample_imgs = fake_imgs[:6]
        #     self.sample_imgs.append(sample_imgs)
        # if len(self.train_imgs) < 6:
        #     # log training images
        #     train_imgs = imgs[:6]
        #     self.train_imgs.append(train_imgs)

        # # Losses
        # self.log_dict(
        #     {
        #         "train/g_loss": error_g.item(),
        #         "train/d_loss": error_d.item()
        #     }, 
        #     prog_bar=True
        # )
        # self.loss_dict["train/g_loss_epoch"].append(error_g.item())
        # self.loss_dict["train/d_loss_epoch"].append(error_d.item())


        imgs, _ = batch
        optimizer_g, optimizer_d = self.optimizers()

        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.model_cfg["latent_dim"], 1, 1)
        z = z.type_as(imgs)

        # Train Generator
        g_loss = self._generator_training_step(imgs, optimizer_g, z)

        # Train Discriminator
        d_loss = 0.
        for _ in range(self.discriminator_updates):
            d_loss += self._discriminator_training_step(imgs, optimizer_d, z)
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


    def configure_optimizers(self):

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
            patience = max(int(0.1*self.hparams.model_cfg["epochs"]), 10)
            scheduler_config_g = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.25, patience=patience, threshold=1e-2, threshold_mode="rel", cooldown=patience),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train/g_loss_epoch",
                "name": "Scheduler-1",
            }
            scheduler_config_d = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.25, patience=patience, threshold=1e-3, threshold_mode="rel", cooldown=patience),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train/d_loss_epoch",
                "name": "Scheduler-2",
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
        self.train_imgs = []


    def on_train_epoch_end(self):

        self.loss_dict["train/g_loss_epoch"] = torch.mean(torch.tensor(self.loss_dict["train/g_loss_epoch"]))
        self.loss_dict["train/d_loss_epoch"] = torch.mean(torch.tensor(self.loss_dict["train/d_loss_epoch"]))
        self.log_dict(
            dictionary=self.loss_dict,
            on_epoch=True,
        )
        
        self._log_images()

        sch = self.lr_schedulers()
        if sch is not None:
            if not isinstance(sch, list):
                sch = [sch]
            for i, scheduler in enumerate(sch):
                if i == 0:
                    # Generator
                    metric = "train/g_loss_epoch"
                else:
                    # Discriminator
                    metric = "train/d_loss_epoch"
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[metric])
                else:
                    scheduler.step()
        

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)