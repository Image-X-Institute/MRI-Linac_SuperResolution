import torch
import numpy as np


def min_max_scaling(
    tensor: torch.Tensor, min_val_desired: float, max_val_desired: float
) -> torch.Tensor:
    """
    Function to perform min-max scaling.

    :param tensor: A Tensor for this scaling calculation to be based off.
    :param min_val_desired: The desired minimum value.
    :param max_val_desired: The desired maximum value.
    """
    min_val_measured = torch.min(tensor)

    max_val_measured = torch.max(tensor)

    fraction = torch.div(tensor - min_val_measured, max_val_measured - min_val_measured)

    rescaled_tensor = torch.mul(fraction, max_val_desired - min_val_desired)

    rescaled_tensor = torch.add(rescaled_tensor, min_val_desired)

    return rescaled_tensor


def min_max_scaling_numpy(
    array: np.ndarray, min_val_desired: float, max_val_desired: float
) -> np.ndarray:
    """
    Function to perform min-max scaling.

    :param array: An array for this scaling calculation to be based off.
    :param min_val_desired: The desired minimum value.
    :param max_val_desired: The desired maximum value.
    """
    min_val_measured = np.amin(array)

    max_val_measured = np.amax(array)

    fraction = np.divide(array - min_val_measured, max_val_measured - min_val_measured)

    rescaled_array = np.multiply(fraction, max_val_desired - min_val_desired)

    rescaled_array = np.add(rescaled_array, min_val_desired)

    return rescaled_array


def retrieve_percentile_scale_val(tensor: torch.Tensor, percentile: int) -> float:
    """
    Function to retrieve a percentile scaling factor to scale
    input data between two arbitrary values (typically 0 and 1).

    :param tensor: A Tensor for this scaling calculation to be based off.
    :param percentile: The percentile to scale to (typically 95).
    """
    scale_val = np.percentile(tensor.cpu().detach().numpy(), percentile)

    return scale_val


def divide_tensor_by_scale_val(tensor: torch.Tensor, scale_val: float) -> torch.Tensor:
    """
    Function to scale a Tensor to a particular scale value.

    :param tensor: A Tensor to be rescaled.
    :param scale_val: The scale value (could be generate from percentile
        scaling for example)
    """
    rescaled_tensor = torch.div(tensor, scale_val)

    return rescaled_tensor


def multiply_tensor_by_scale_val(
    tensor: torch.Tensor, scale_val: float
) -> torch.Tensor:
    """
    Function to scale a Tensor to a particular scale value.

    :param tensor: A Tensor to be rescaled.
    :param scale_val: The scale value (could be generate from percentile
        scaling for example)
    """
    rescaled_tensor = torch.mul(tensor, scale_val)

    return rescaled_tensor
