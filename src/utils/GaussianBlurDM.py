import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset


def blur_image(image, kernel_size, sigma):
    return GaussianBlur(kernel_size, sigma)(image)


def multiple_blur_image(image, kernel_size, sigma, iters):
    for i in range(iters):
        image = blur_image(image, kernel_size, sigma)

    return image


def batch_blur(batch, timestep, kernel_size, sigma):
    return torch.stack(
        [
            multiple_blur_image(batch[i], kernel_size, sigma, timestep[i])
            for i in range(batch.shape[0])
        ]
    )


class GaussianBlurDM(nn.Module):
    def __init__(
        self,
        gt: nn.Module,
        kernel_size: int,
        sigma: float,
        n_T: int,
        mnist_dataset: MNIST,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        # this is the decoder model that will be used to estimate the diffusion / encoder process
        self.gt = gt
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.n_T = n_T
        self.criterion = criterion

        self.mnist_dataset = mnist_dataset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sample a random time step t for each batch element
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        z_t = batch_blur(x, t, self.kernel_size, self.sigma)

        # Now we predict the error term using the model, we pass in the latent variable
        # z_t and the time step t to the model, and get the model to try and predict eps.
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

        x, _ = next(iter(dataloader))
        _one = torch.ones(n_sample, device=device)

        # we generate z_T by blurring some initial MNIST images n_T times.
        z_t = (
            batch_blur(x, [self.n_T] * n_sample, self.kernel_size, self.sigma)
            .float()
            .to(device)
        )

        for t in range(self.n_T, 0, -1):
            x_0_pred = self.gt(z_t, (t / self.n_T) * _one)
            z_t = (
                z_t
                - batch_blur(x_0_pred, [t] * n_sample, self.kernel_size, self.sigma)
                + batch_blur(x_0_pred, [t - 1] * n_sample, self.kernel_size, self.sigma)
            )

        return z_t
