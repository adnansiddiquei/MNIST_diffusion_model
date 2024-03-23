from utils import create_dir_if_required, DiffusionModelTrainer, GaussianBlurDM, CNN, DDPM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def main():
    output_dir = create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(f'{output_dir}/outputs', 'ddpm')

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4,
                            drop_last=True)

    # train the default model
    gt_1 = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
    ddpm_1 = DDPM(gt=gt_1, betas=(1e-4, 0.02), n_T=1000)
    trainer = DiffusionModelTrainer(ddpm_1, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/default_model')

    # train the short model
    gt_2 = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
    ddpm_2 = DDPM(gt=gt_2, betas=(1e-4, 0.1), n_T=200)
    trainer = DiffusionModelTrainer(ddpm_2, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/short_model')
    
    # train the long model
    gt_3 = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
    ddpm_3 = DDPM(gt=gt_3, betas=(1e-4, 0.004), n_T=5000)
    trainer = DiffusionModelTrainer(ddpm_3, dataloader)
    trainer.train(n_epoch=100, save_folder=f'{output_dir}/long_model')


if __name__ == '__main__':
    main()
