import torch
from tqdm import tqdm
from ._load_model import load_model


def generate_samples(model_name: str, epoch_range: range | list, num_samples, device):
    all_samples = torch.zeros(len(epoch_range), num_samples, 1, 28, 28)

    with torch.no_grad():
        for idx, epoch in enumerate(tqdm(epoch_range)):
            # generate samples
            ddpm = load_model(model_name, epoch).to(device)

            samples = ddpm.sample(num_samples, (1, 28, 28), device)
            samples = samples.to('cpu')

            all_samples[idx] = samples

            del ddpm
            del samples

    return all_samples
