import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from utils import (
    save_pickle,
    load_pickle,
    calc_loss_per_epoch,
    find_latest_model,
    save_images,
)


class DiffusionModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optim: torch.optim.Optimizer = None,
        accelerator: Accelerator = None,
    ):
        """
        Trainer for a DDPM and FashionMNIST model.

        Parameters
        ----------
        model : nn.Module
            The model to train. Either a DDPM or FashionMNIST model.
        dataloader : DataLoader
            The dataloader to use for training the model.
        optim : torch.optim.Optimizer
            The optimizer to use for training the model. If None, Adam with a learning rate of 2e-4 is used.
        accelerator : Accelerator
            The accelerator to use for training the model. If None, a new accelerator is created.
        """
        self.model = model
        self.optim = optim
        self.dataloader = dataloader
        self.accelerator = accelerator
        self.losses = []

        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        if self.accelerator is None:
            self.accelerator = Accelerator()

        self.model, self.optim, self.dataloader = self.accelerator.prepare(
            self.model, self.optim, self.dataloader
        )

    def train(self, n_epoch, save_folder, save_init_images=False):
        """
        Train the model for n_epoch epochs.

        Parameters
        ----------
        n_epoch : int
            The number of epochs to train the model for.
        save_folder : str
            The folder to save the model and losses to.
        save_init_images : bool
            Whether to save the initial images when sampling from the model. Defaults to false, and this will only
            do anything for FashionMNISTDM models.

        Returns
        -------
        None
        """
        # load the latest trained model and losses, if this is a second run then the model will be loaded
        # and the training continues from the last saved epoch
        latest_model, latest_model_epoch = find_latest_model(save_folder)

        if latest_model is not None:
            self.model.load_state_dict(latest_model)
            self.losses = load_pickle(f'{save_folder}/losses_batch.pkl')

            print(
                f'Successfully loaded model {latest_model_epoch} and losses from previous training session.'
            )

        # start the training loop for however many epochs are specified
        for i in range(latest_model_epoch + 1, n_epoch):
            self.model.train()

            # train for 1 epoch
            pbar = tqdm(self.dataloader)  # Wrap our loop with a visual progress bar
            for x, _ in pbar:
                self.optim.zero_grad()

                loss = self.model(x)

                loss.backward()
                # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

                self.losses.append(loss.item())

                avg_loss = np.average(self.losses[min(len(self.losses) - 100, 0) :])

                pbar.set_description(
                    f'Epoch {i} -- loss: {avg_loss:.3g}'
                )  # Show running average of loss in progress bar

                self.optim.step()

            self.model.eval()

            # Now do some sampling and save the model and losses
            with torch.no_grad():
                if save_init_images:
                    xh, init_images = self.model.sample(
                        16,
                        (1, 28, 28),
                        self.accelerator.device,
                        return_initial_images=True,
                    )

                    save_images(init_images, 4, f'{save_folder}/sample_init{i:04d}.png')
                else:
                    (xh,) = self.model.sample(16, (1, 28, 28), self.accelerator.device)

                save_images(xh, 4, f'{save_folder}/sample_{i:04d}.png')

                # save model_name
                torch.save(self.model.state_dict(), f'{save_folder}/model_{i}.pth')

            save_pickle(self.losses, f'{save_folder}/losses_batch.pkl')
            save_pickle(
                calc_loss_per_epoch(self.losses), f'{save_folder}/losses_epoch.pkl'
            )
