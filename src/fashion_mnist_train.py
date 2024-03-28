from utils import create_dir_if_required, DiffusionModelTrainer, CNN, FashionMNISTDM
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def main():
    output_dir = create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(f'{output_dir}/outputs', 'fashion_mnist')

    train_dataset = MNIST(
        './data', train=True, download=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
    )

    gt = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )

    diffusion_model = FashionMNISTDM(gt, betas=(1e-4, 0.02), n_T=1000)

    trainer = DiffusionModelTrainer(diffusion_model, dataloader)

    trainer.train(n_epoch=100, save_folder=output_dir, save_init_images=True)


if __name__ == '__main__':
    main()
