import torch
from tqdm import tqdm
from ._load_model import load_model


def generate_samples(model_name: str, epoch_range: range | list, num_samples, device):
    """
    Generate samples from the model_name at the specified epochs.

    Parameters
    ----------
    model_name : str
        The name of the model to generate samples from. The name of a model is the path to the model folder starting
        at `src/outputs`. So the name of the default ddpm model would be 'ddpm/default_model'.
    epoch_range : range | list
        The epochs to generate samples from.
    num_samples
        The number of samples to generate per epoch.
    device : str
        The device to run the model on.

    Returns
    -------
    torch.Tensor
        A tensor of shape (len(epoch_range), num_samples, 1, 28, 28) containing the generated samples.
    """
    all_samples = torch.zeros(len(epoch_range), num_samples, 1, 28, 28)

    with torch.no_grad():
        for idx, epoch in enumerate(tqdm(epoch_range)):
            # generate samples
            model = load_model(model_name, epoch).to(device)

            if model_name == 'fashion_mnist':
                model.set_sample_size(num_samples)

            samples = model.sample(num_samples, (1, 28, 28), device)
            samples = samples.to('cpu')

            all_samples[idx] = samples

            del model
            del samples

    return all_samples
