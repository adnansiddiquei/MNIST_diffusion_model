import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset


# def blur_image(image, kernel_size, sigma):
#     return GaussianBlur(kernel_size, sigma)(image)
#
#
# def multiple_blur_image(image, kernel_size, sigma, iters):
#     for i in range(iters):
#         image = blur_image(image, kernel_size, sigma)
#
#     return image
#
#
# def batch_blur(batch, timestep, kernel_size, sigma):
#     return torch.stack(
#         [
#             multiple_blur_image(batch[i], kernel_size, sigma, timestep[i])
#             for i in range(batch.shape[0])
#         ]
#     )


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
        self, n_sample: int, size, device, checkpoints: list = None
    ) -> torch.Tensor:
        dataloader = DataLoader(
            Subset(self.mnist_dataset, indices=range(n_sample * 2)),
            batch_size=n_sample,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        # get a batch of images
        x, _ = next(iter(dataloader))
        _one = torch.ones(n_sample, device=device)

        # we generate z_T by blurring some initial MNIST images n_T times. This gives us a reasonable starting point
        # to start the decoding process from.
        z_t = (
            batch_blur(x, self.sigma_schedule[-1].repeat(n_sample), self.kernel_size)
            .float()
            .to(device)
        )

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

        return z_t
