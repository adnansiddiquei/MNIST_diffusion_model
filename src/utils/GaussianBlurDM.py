import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import numpy as np


def batch_blur_random(batch, kernel_size, iterations):
    """Randomly blur an image in a batch multiple times."""

    def multiple_blur_image(image):
        for i in range(iterations):
            image = GaussianBlur(kernel_size, np.random.normal(3, 1))(image)

        return image

    return torch.stack([multiple_blur_image(batch[i]) for i in range(batch.shape[0])])


def batch_blur(
    x: torch.Tensor, sigmas: torch.Tensor, kernel_size: int = 29
) -> torch.Tensor:
    """
    Apply Gaussian blur to a batch of images.


    Parameters
    ----------
    x : torch.Tensor
        A batch of images to be blurred. Shape (batch_size, channels, H, W).
    sigmas : torch.Tensor
        The standard deviation of the Gaussian blur to be applied to each image in the batch. Shape (batch_size,).
    kernel_size : int
        The size of the kernel to be used in the Gaussian blur. Default is 29.

    Returns
    -------
    torch.Tensor
        The blurred images. Shape (batch_size, channels, H, W).
    """
    return torch.stack(
        [GaussianBlur(kernel_size, sigmas[i].item())(x[i]) for i in range(x.shape[0])]
    )


def gaussian_blur_schedule(
    sigma_min: float, sigma_max: float, n_T: int
) -> torch.Tensor:
    """
    Create a schedule of standard deviations for Gaussian blur.

    Parameters
    ----------
    sigma_min : float
        The minimum standard deviation.
    sigma_max : float
        The maximum standard deviation.
    n_T : int
        The number of time steps.

    Returns
    -------
    torch.Tensor
        The schedule of standard deviations. Shape (n_T,).
    """
    return torch.linspace(sigma_min, sigma_max, n_T)


class GaussianBlurDM(nn.Module):
    def __init__(
        self,
        gt: nn.Module,
        kernel_size: int,
        sigma_range: tuple[float, float],
        n_T: int,
        mnist_dataset: MNIST,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        # this is the decoder model_name that will be used to estimate the diffusion / encoder process
        self.gt = gt
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.n_T = n_T
        self.criterion = criterion

        self.mnist_dataset = mnist_dataset

        self.register_buffer(
            'sigma_schedule',
            gaussian_blur_schedule(sigma_range[0], sigma_range[1], n_T),
        )

        self.sigma_schedule  # et by register_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sample a random time step t for each batch element
        t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)

        # get the sigma for the gaussian blue at the corresponding time step t
        sigma_t = self.sigma_schedule[t]

        # blur each image in the batch x with the corresponding blur
        z_t = batch_blur(x, sigma_t, self.kernel_size)

        # get the CNN to try and predict the original image x from the blurred image z_t
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
            dataloader = DataLoader(
                Subset(self.mnist_dataset, indices=range(n_sample * 2)),
                batch_size=n_sample,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )

            # get a batch of images
            x, _ = next(iter(dataloader))
        else:
            assert initial_images.shape[0] == n_sample
            assert initial_images.shape[1] == size[0]
            assert initial_images.shape[2] == size[1]
            assert initial_images.shape[3] == size[2]

            x = initial_images

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
            batch_blur(x, self.sigma_schedule[-1].repeat(n_sample), self.kernel_size)
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
                sigma_t = self.sigma_schedule[t]
                sigma_t_minus_1 = self.sigma_schedule[t - 1]

                z_t = (
                    z_t
                    - batch_blur(x_0_pred, sigma_t.repeat(n_sample), self.kernel_size)
                    + batch_blur(
                        x_0_pred, sigma_t_minus_1.repeat(n_sample), self.kernel_size
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
            return checkpoints, x
        elif checkpoints and not return_initial_images:
            return checkpoints
        elif not checkpoints and return_initial_images:
            return z_t, x
        else:
            return z_t
