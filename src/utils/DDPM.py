"""
Here we define the actual diffusion model_name, which specifies the training
schedule, takes an arbitrary model_name for estimating the
diffusion process (such as the CNN above),
and computes the corresponding loss (as well as generating samples).
"""

import torch
import torch.nn as nn
from typing import Tuple
from .utils import ddpm_schedules


class DDPM(nn.Module):
    def __init__(
        self,
        gt: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """
        Denoising Diffusion Probabilistic Model (DDPM) class.

        Used to train and sample from a diffusion model_name.

        Parameters
        ----------
        gt : nn.Module
            The model_name to use for estimating the diffusion process. This model_name should take in the latent variable z_t
            and the time step t, and output the predicted error term.
        betas : Tuple[float, float]
         A tuple specifying the range of beta values for the noise schedule. Betas control the amount of noise added
         at each step of the diffusion process.
        n_T : int
         The number of steps in the diffusion process.
        criterion : nn.Module
         The loss function to use. Default, this is the mean squared error loss.
        """
        super().__init__()

        # this is the decoder model_name that will be used to estimate the diffusion / encoder process
        self.gt = gt

        # Create the noise schedule, which is a dictionary of alpha_t and beta_t values
        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model_name parameters. This is useful for constants.
        self.register_buffer('beta_t', noise_schedule['beta_t'])
        self.beta_t  # Exists! Set by register_buffer

        self.register_buffer('alpha_t', noise_schedule['alpha_t'])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 18.1 in Prince. Forward pass of the DDPM model_name.

        Parameters
        ----------
        x : torch.Tensor
         The input data to the model_name. Has shape...
        """
        # sample a random time step t for each batch element
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        # Sample standard normal noise, with the same shape as x
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        # Get the alpha_t value corresponding to the time step t
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        # (Eqn. 18.7, Prince) This is the latent variable z_t, it is the input x with some noise added
        # equivalent to the amount of noise that should be added at time t
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps

        # Now we predict the error term using the model_name, we pass in the latent variable
        # z_t and the time step t to the model_name, and get the model_name to try and predict eps.
        preds = self.gt(z_t, t / self.n_T)

        return self.criterion(eps, preds)

    def sample(
        self, n_sample: int, size, device, checkpoints: list = None
    ) -> torch.Tensor:
        """
        Algorithm 18.2 in Prince.

        Generates samples from noise using the reverse diffusion process learned by the model_name.

        Parameters
        ----------
        n_sample : int
         The number of samples to generate.
        size : Tuple
         The size of the samples to generate. Shape is (channels, height, width).
        device : torch.device
            The device to use for the samples.
        checkpoints : list
            A list of time steps to return the latent variables at, as well as the initial and final samples.
            E.g., this can be provided as [750, 500, 250] and the final tensor will be of shape
            (len(checkpoints) + 2, n_sample, *size) where the extra 2 are the initial and final samples,
            with the rest being the latent variables at the specified time steps.

        Returns
        -------
        torch.Tensor
         The generated samples (with the checkpoints if provided).
        """
        # create a checkpoints tensor, if required, to store the latent variable at the specified time steps
        checkpoints = list(checkpoints) if checkpoints else None
        checkpoint_tensors = (
            torch.empty(n_sample, len(checkpoints) + 2, *size, device=device)
            if checkpoints
            else None
        )

        # Create a tensor of ones with the same shape as the number of samples
        _one = torch.ones(n_sample, device=device)

        # Sample standard normal noise
        z_t = torch.randn(n_sample, *size, device=device)

        # save the initial samples into checkpoint_tensors, if required
        if checkpoints:
            for i in range(n_sample):
                checkpoint_tensors[i, 0] = z_t[i]

        # Iterate backwards through the noise schedule
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]  # Get the alpha_t value for the current time step
            beta_t = self.beta_t[i]  # Get the beta_t value for the current time step

            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            # if this iteration is a checkpoint value, save the latent variable into the checkpoint tensor
            if checkpoints:
                if i in checkpoints:
                    for j in range(n_sample):
                        checkpoint_tensors[j, checkpoints.index(i) + 1] = z_t[j]

            if i > 1:
                # Last line of loop
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        # save the final samples into checkpoint_tensors, if required
        if checkpoints:
            for i in range(n_sample):
                checkpoint_tensors[i, -1] = z_t[i]

        return z_t if checkpoints is None else checkpoint_tensors
