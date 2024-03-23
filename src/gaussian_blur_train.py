from utils import create_dir_if_required, DiffusionModelTrainer, GaussianBlurDM, CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset


def main():
    output_dir = create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(f'{output_dir}/outputs', 'gaussian_blur')

    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    # dataloader = DataLoader(Subset(dataset, indices=range(128*100)), batch_size=128, shuffle=True, num_workers=4,
    #                         drop_last=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4,
                            drop_last=True)

    gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)

    diffusion_model = GaussianBlurDM(gt, kernel_size=11, sigma=7, n_T=20, mnist_dataset=dataset)

    trainer = DiffusionModelTrainer(diffusion_model, dataloader)

    trainer.train(n_epoch=100, save_folder=output_dir)


if __name__ == '__main__':
    main()
