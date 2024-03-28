import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import numpy as np
from utils import save_pickle, load_pickle, calc_loss_per_epoch, find_latest_model


class DiffusionModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optim: torch.optim.Optimizer = None,
        accelerator: Accelerator = None,
    ):
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

    def train(self, n_epoch, save_folder):
        latest_model, latest_model_epoch = find_latest_model(save_folder)

        if latest_model is not None:
            self.model.load_state_dict(latest_model)
            self.losses = load_pickle(f'{save_folder}/losses_batch.pkl')

            print(
                f'Successfully loaded model {latest_model_epoch} and losses from previous training session.'
            )

        for i in range(latest_model_epoch + 1, n_epoch):
            self.model.train()

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

            with torch.no_grad():
                xh = self.model.sample(
                    16, (1, 28, 28), self.accelerator.device
                )  # Can get device explicitly with `accelerator.device`
                grid = make_grid(xh, nrow=4)

                # Save samples to `./contents` directory
                save_image(grid, f'{save_folder}/sample_{i:04d}.png')

                # save model_name
                torch.save(self.model.state_dict(), f'{save_folder}/model_{i}.pth')

            save_pickle(self.losses, f'{save_folder}/losses_batch.pkl')
            save_pickle(
                calc_loss_per_epoch(self.losses), f'{save_folder}/losses_epoch.pkl'
            )
