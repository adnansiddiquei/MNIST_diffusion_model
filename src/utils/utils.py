from typing import Dict
import torch
import pickle
import os


def create_dir_if_required(script_filepath: str, dir_name: str) -> str:
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    dir_to_make = os.path.join(cwd, dir_name)

    if not os.path.exists(os.path.join(cwd, dir_name)):
        os.makedirs(dir_to_make)

    return dir_to_make


def save_pickle(obj: object, path: str) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    obj : object
        The Python object to save.
    path : str
        The path to save the object to.

    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        The path to the pickle file.

    Returns
    -------
    object
        The Python object loaded from the pickle file.

    """
    with open(path, "rb") as f:
        return pickle.load(f)


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling with a linear noise schedule.

    Noise schedule increases linearly from `beta1` to `beta2` over `T` timesteps.
    The `alpha_t` schedule is computed as the cumulative product of `(1 - beta_t)`.

    This function is used for evaluating and training the diffusion model.


    Parameters
    ----------
    beta1 : float
        The initial value of the noise schedule. Must be in (0, 1).
    beta2 : float
        The final value of the noise schedule. Must be in (0, 1).
    T : int
        The number of time-steps to use in the diffusion / encoder process.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing the noise schedule `beta_t` and the corresponding `alpha_t` schedule.
        Both schedules are returned as tensors of shape (T+1,).

    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Create a linear schedule from `beta1` to `beta2` over `T` timesteps
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1

    # Compute the corresponding `alpha_t` schedule (Sec 18.2.1, Eqn, 18.7; Prince)
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}
