import torch

from oak_generation.models.custom_models.DCGAN.DCGAN import DCGAN


class WGAN(DCGAN):
    """
    Custom Implementation of Wasserstein GAN with Gradient Penalty 
    """

    def __init__(self, model_cfg, **kwargs):
        super(WGAN, self).__init__(model_cfg, **kwargs)

        # Weight Coefficient of Gradient Penalty
        self.lambda_gp = 10.
        # Update Discriminator Multiple Times per Iteration
        self.discriminator_updates = 5

    def _gradient_penalty(self, real_imgs, fake_imgs, labels=None):
        """
        Function to compute gradient penalty as presented in WGAN-GP Paper
        """
        alpha = torch.rand(len(real_imgs), 1, 1, 1, device=self.hparams.model_cfg["device"], requires_grad=True)

        mixed_imgs = alpha * real_imgs + (1 - alpha) * fake_imgs

        # Used to handle conditional inputs
        if labels is not None:
            d_mixed = self.discriminator(mixed_imgs, labels)
        else:
            d_mixed = self.discriminator(mixed_imgs)

        # Note: You need to take the gradient of outputs with respect to inputs.
        mixed_gradient = torch.autograd.grad(
            inputs = mixed_imgs,
            outputs = d_mixed,
            grad_outputs=torch.ones_like(d_mixed),
            create_graph=True,
            retain_graph=True,
        )[0]

        mixed_gradient = mixed_gradient.view(len(mixed_gradient), -1)
        mixed_gradient_norm = mixed_gradient.norm(2, dim=1)
        return torch.mean(torch.square(mixed_gradient_norm - 1))

    def _generator_loss(self, z, valid):
        """
        WGAN with Gradient Penalty Loss Function
        """
        fake_imgs = self(z)
        return -1. * torch.mean(self.discriminator(fake_imgs))

    def _discriminator_loss(self, imgs, valid, z, fake):
        """
        WGAN with Gradient Penalty Loss Function
        """
        d_real = self.discriminator(imgs).view(-1)
        fake_imgs = self(z).detach()
        d_fake = self.discriminator(fake_imgs).view(-1)
        return torch.mean(d_fake - d_real + self.lambda_gp * self._gradient_penalty(imgs, fake_imgs))
