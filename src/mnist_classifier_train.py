from utils import (
    create_dir_if_required,
    CNNClassifier,
    save_pickle,
    find_latest_model,
    load_pickle,
    calc_loss_per_epoch,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np


def main():
    create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(__file__, 'outputs/mnist_classifier')

    tf = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST('./data', train=True, download=True, transform=tf)
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
    )

    test_dataset = MNIST('./data', train=False, download=True, transform=tf)
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True
    )

    model = CNNClassifier(
        1, (32, 64, 128, 64), 10, adaptive_pooling_output_size=(4, 4)
    )  # epoch 7, 99.23% test accuracy

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    accelerator = Accelerator()

    model, optim, train_dataloader, test_dataloader = accelerator.prepare(
        model, optim, train_dataloader, test_dataloader
    )

    n_epoch = 100
    train_loss_batch = []
    test_loss_batch = []

    latest_model, latest_model_epoch = find_latest_model(output_dir)

    if latest_model is not None:
        model.load_state_dict(latest_model)
        train_loss_batch = load_pickle(f'{output_dir}/train_losses_batch.pkl')
        test_loss_batch = load_pickle(f'{output_dir}/test_losses_batch.pkl')

        print(
            f'Successfully loaded model {latest_model_epoch} and losses from previous training session.'
        )

    for i in range(n_epoch):
        model.train()

        pbar = tqdm(train_dataloader)  # Wrap our loop with a visual progress bar

        for x, label in pbar:
            optim.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, label)
            loss.backward()
            train_loss_batch.append(loss.item())

            avg_loss = np.average(train_loss_batch[-100:])
            pbar.set_description(
                f'Epoch {i} --- Train loss: {avg_loss:.3g}'
            )  # Show running average of loss in progress bar

            optim.step()

        model.eval()
        correct_count = 0

        for x, label_test in test_dataloader:
            with torch.no_grad():
                preds_test = model(x)
                loss_test = loss_fn(preds_test, label_test)
                test_loss_batch.append(loss_test.item())

                # Calculate number of correct predictions
                _, predicted = torch.max(preds_test.data, 1)
                correct_count += (predicted == label_test).sum().item()

        # Calculate accuracy
        accuracy = correct_count / len(test_dataset)

        print(
            f'Epoch {i}, Test Loss: {np.mean(test_loss_batch[-78:]):.4f}, Accuracy: {accuracy:.2%}'
        )

        torch.save(model.state_dict(), f'{output_dir}/model_{i}.pth')

        # save the losses
        save_pickle(train_loss_batch, f'{output_dir}/train_losses_batch.pkl')
        save_pickle(test_loss_batch, f'{output_dir}/train_losses_batch.pkl')

        save_pickle(
            calc_loss_per_epoch(train_loss_batch),
            f'{output_dir}/train_losses_epoch.pkl',
        )

        save_pickle(
            calc_loss_per_epoch(test_loss_batch, 78),
            f'{output_dir}/test_losses_epoch.pkl',
        )


if __name__ == '__main__':
    main()
