import torch
import torch.nn as nn
from typing import Tuple
from .utils import ddpm_schedules
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)


class FashionMNISTDM(nn.Module):
    def __init__(
        self,
        gt: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
        train_batch_size: int = 128,
        sample_size: int = 16,
    ) -> None:
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

        self.train_dataset = FashionMNIST(
            f'{current_dir}/../../data', download=True, train=True, transform=ToTensor()
        )
        self.test_dataset = FashionMNIST(
            f'{current_dir}/../../data',
            download=True,
            train=False,
            transform=ToTensor(),
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.sample_dataloader = None
        self.sample_size = None

        self.set_sample_size(sample_size)

    def set_sample_size(self, n_sample: int):
        self.sample_size = n_sample
        self.sample_dataloader = DataLoader(
            self.test_dataset,
            batch_size=n_sample,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

    def encode(
        self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sample a random time step t for each batch element
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        eps, _ = next(iter(self.train_dataloader))
        eps = eps.to(x.device)
        z_t = self.encode(x, t, eps)

        preds = self.gt(z_t, t / self.n_T)

        return self.criterion(x, preds)

    def sample(
        self,
        n_sample: int,
        size,
        device,
        checkpoints: list = None,
        return_initial_images: bool = False,
        initial_images: torch.Tensor = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if initial_images is None:
            # get a batch of images
            x, _ = next(iter(self.sample_dataloader))
            x = x.to(device)
        else:
            assert initial_images.shape[0] == n_sample
            assert initial_images.shape[1] == size[0]
            assert initial_images.shape[2] == size[1]
            assert initial_images.shape[3] == size[2]

            x = initial_images.to(device)

        _one = torch.ones(n_sample, device=device)

        # create a checkpoints tensor, if required, to store the latent variable at the specified time steps
        checkpoints = list(checkpoints) if checkpoints else None
        checkpoint_tensors = (
            torch.empty(n_sample, len(checkpoints) + 2, *size, device=device)
            if checkpoints
            else None
        )

        # we generate z_T by blurring some initial MNIST images n_T times. This gives us a reasonable starting point
        # to start the decoding process from.
        z_t = (
            self.encode(
                x, torch.tensor(self.n_T - 1, device=device).repeat(n_sample), eps=x
            )
            .float()
            .to(device)
        )

        # save the initial samples into checkpoint_tensors, if required
        if checkpoints:
            for i in range(n_sample):
                checkpoint_tensors[i, 0] = z_t[i]

        # Algorithm 2 from "Denoising Diffusion Probabilistic Models", Ho, et al. 2020.
        for t in reversed(range(0, self.n_T)):
            x_0_pred = self.gt(z_t, (t / self.n_T) * _one)

            if t > 0:
                z_t = (
                    z_t
                    - self.encode(
                        x_0_pred, torch.tensor(t, device=device).repeat(n_sample), eps=x
                    )
                    + self.encode(
                        x_0_pred,
                        torch.tensor(t - 1, device=device).repeat(n_sample),
                        eps=x,
                    )
                )
            else:
                z_t = x_0_pred

            # if this iteration is a checkpoint value, save the latent variable into the checkpoint tensor
            if checkpoints:
                if t in checkpoints:
                    for j in range(n_sample):
                        checkpoint_tensors[j, checkpoints.index(t) + 1] = z_t[j]

        # save the final samples into checkpoint_tensors, if required
        if checkpoints:
            for i in range(n_sample):
                checkpoint_tensors[i, -1] = z_t[i]

        if checkpoints and return_initial_images:
            return checkpoint_tensors, x
        elif checkpoints and not return_initial_images:
            return checkpoint_tensors
        elif not checkpoints and return_initial_images:
            return z_t, x
        else:
            return z_t
