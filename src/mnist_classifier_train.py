from utils import create_dir_if_required, CNNClassifier, save_pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np


def main():
    output_dir = create_dir_if_required(__file__, 'outputs')
    output_dir = create_dir_if_required(f'{output_dir}/outputs', 'mnist_classifier')

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

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
    ddpm, optim, train_dataloader, test_dataloader = accelerator.prepare(
        model, optim, train_dataloader, test_dataloader
    )

    n_epoch = 100
    train_loss = []
    test_loss = []

    for i in range(n_epoch):
        model.train()

        pbar = tqdm(train_dataloader)  # Wrap our loop with a visual progress bar

        for x, label in pbar:
            optim.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, label)
            loss.backward()
            train_loss.append(loss.item())

            avg_loss = np.average(train_loss[-100:])
            pbar.set_description(
                f'loss: {avg_loss:.3g}'
            )  # Show running average of loss in progress bar

            optim.step()

        model.eval()
        test_loss_batch = []
        correct_count = 0

        for x, label_test in test_dataloader:
            with torch.no_grad():
                preds_test = model(x)
                loss_test = loss_fn(preds_test, label_test)
                test_loss_batch.append(loss_test.item())

                # Calculate number of correct predictions
                _, predicted = torch.max(preds_test.data, 1)
                correct_count += (predicted == label_test).sum().item()

        avg_test_loss = np.average(test_loss_batch)
        test_loss.append(avg_test_loss)

        # Calculate accuracy
        accuracy = correct_count / len(test_dataset)

        print(
            f'Epoch [{i + 1}/{n_epoch}], Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2%}'
        )

        torch.save(ddpm.state_dict(), f'{output_dir}/model_{i}.pth')

    save_pickle(train_loss, f'{output_dir}/train_loss.pkl')
    save_pickle(test_loss, f'{output_dir}/train_loss.pkl')


if __name__ == '__main__':
    main()
