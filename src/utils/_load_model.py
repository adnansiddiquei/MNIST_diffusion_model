import os

import torch

from . import CNN, DDPM, GaussianBlurDM, CNNClassifier
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def load_model(model_name, epoch):
    current_file = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file)

    ddpm_path = os.path.join(current_dir, '../outputs/ddpm')
    gaussian_blur_path = os.path.join(current_dir, '../outputs/gaussian_blur')
    mnist_classifier_path = os.path.join(current_dir, '../outputs/mnist_classifier')
    root_path = os.path.join(current_dir, '../..')

    model_name = model_name.split('/')

    if model_name[0] == 'ddpm' and len(model_name) == 2:
        gt = CNN(
            in_channels=1,
            expected_shape=(28, 28),
            n_hidden=(16, 32, 64, 32, 16),
            act=nn.GELU,
        )

        if model_name[1] == 'default_model':
            model = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)
            state_dict = torch.load(f'{ddpm_path}/default_model/model_{epoch}.pth')
            model.load_state_dict(state_dict)
        elif model_name[1] == 'long_model':
            model = DDPM(gt=gt, betas=(1e-4, 0.004), n_T=5000)
            state_dict = torch.load(f'{ddpm_path}/long_model/model_{epoch}.pth')
            model.load_state_dict(state_dict)
        elif model_name[1] == 'short_model':
            model = DDPM(gt=gt, betas=(1e-4, 0.1), n_T=200)
            state_dict = torch.load(f'{ddpm_path}/short_model/model_{epoch}.pth')
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f'Unknown model: {model_name}')
    elif model_name[0] == 'guassian_blur':
        test_dataset = MNIST(
            f'{root_path}/data',
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        gt = CNN(
            in_channels=1,
            expected_shape=(28, 28),
            n_hidden=(16, 32, 64, 32, 16),
            act=nn.GELU,
        )
        model = GaussianBlurDM(
            gt,
            kernel_size=29,
            sigma_range=(0.5, 20),
            n_T=20,
            mnist_dataset=test_dataset,
        )
        state_dict = torch.load(f'{gaussian_blur_path}/model_{epoch}.pth')
        model.load_state_dict(state_dict)
    elif model_name[0] == 'mnist_classifier':
        model = CNNClassifier(
            1, (32, 64, 128, 64), 10, adaptive_pooling_output_size=(4, 4)
        )
        state_dict = torch.load(f'{mnist_classifier_path}/model_{epoch}.pth')
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return model
