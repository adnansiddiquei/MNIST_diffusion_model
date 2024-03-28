"""
Train the 3 DDPM models on the MNIST dataset for 100 epochs each.

The 3 models are trained with different noise schedules, as specified in the main function report.
"""

from utils import (
    create_dir_if_required,
    DiffusionModelTrainer,
    CNN,
    DDPM,
)
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def main():
    # Create all the output directories
    output_dir = create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(__file__, 'outputs/ddpm')
    create_dir_if_required(__file__, 'outputs/ddpm/default_model')
    create_dir_if_required(__file__, 'outputs/ddpm/short_model')
    create_dir_if_required(__file__, 'outputs/ddpm/long_model')

    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./data', train=True, download=True, transform=tf)
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
    )

    # train the default model
    gt_1 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )
    ddpm_1 = DDPM(gt=gt_1, betas=(1e-4, 0.02), n_T=1000)
    trainer = DiffusionModelTrainer(ddpm_1, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/default_model')

    # train the short model
    gt_2 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )
    ddpm_2 = DDPM(gt=gt_2, betas=(1e-4, 0.1), n_T=200)
    trainer = DiffusionModelTrainer(ddpm_2, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/short_model')

    # train the long model
    gt_3 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )
    ddpm_3 = DDPM(gt=gt_3, betas=(1e-4, 0.004), n_T=5000)
    trainer = DiffusionModelTrainer(ddpm_3, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/long_model')


if __name__ == '__main__':
    main()
