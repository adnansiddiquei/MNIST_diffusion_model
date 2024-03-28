import os

import torch

from .CNN import CNNClassifier, CNN
from .DDPM import DDPM
from .FashionMNISTDM import FashionMNISTDM


import torch.nn as nn


def load_model(model_name, epoch):
    current_file = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file)

    ddpm_path = os.path.join(current_dir, '../outputs/ddpm')
    fashion_mnist_path = os.path.join(current_dir, '../outputs/fashion_mnist')
    mnist_classifier_path = os.path.join(current_dir, '../outputs/mnist_classifier')
    # root_path = os.path.join(current_dir, '../..')

    gt = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )

    model_name = model_name.split('/')

    if model_name[0] == 'ddpm' and len(model_name) == 2:
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
    elif model_name[0] == 'fashion_mnist':
        model = FashionMNISTDM(gt, betas=(1e-4, 0.02), n_T=1000)
        state_dict = torch.load(f'{fashion_mnist_path}/model_{epoch}.pth')
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
