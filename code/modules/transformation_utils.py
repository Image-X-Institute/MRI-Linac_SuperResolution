import torch
import numpy as np


def convert_numpy2tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Performs conversion of Numpy array on the CPU to Torch Tensor
    on the GPU.

    :param array: A Numpy array.
    :param device: device to run inference on.
    """
    return torch.from_numpy(array).to(device)


def convert_tensor2numpy(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs conversion of Torch Tensor on the GPU to Numpy array
    on the CPU.

    :param tensor: A torch Tensor.
    """
    return tensor.cpu().detach().numpy()


def convert2rgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    Method to convert an initially grayscale tensor to RGB.

    :param tensor: A torch Tensor.
    """
    rgb_tensor = torch.stack((0.33 * tensor, 0.33 * tensor, 0.33 * tensor), dim=1)

    return rgb_tensor


def squeeze(tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove all singleton dimensions.

    :param tensor: A torch Tensor with singleton dimensions.
    """
    return torch.squeeze(tensor)


def unsqueeze(tensor: torch.Tensor) -> torch.Tensor:
    """
    Add a singleton dimension at the first position.

    :param tensor: A torch Tensor of any shape.
    """
    return torch.unsqueeze(tensor, 0)


def rotate(tensor: torch.Tensor, num_rotation: int) -> torch.Tensor:
    """
    Rotate a tensor 90 degrees a number of times.
    Works most intuitively with 2D tensors.

    :param tensor: A torch Tensor of any shape.
    :param num_rotation: Number of 90 degree rotations.
    """
    return torch.rot90(tensor, num_rotation)


def get_magnitude_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Retrieves the magnitude Tensors, caution if doing this
    to k-space. Required for most neural network applications
    as data is sometimes saved in Gadgetron as complex data.

    :param tensor: A torch Tensor.
    """
    return torch.abs(tensor)


def reshape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Reshaped a tensor to a specified shape.

    :param tensor: A torch Tensor.
    :param shape: A shape to be reshaped to.
    """

    tensor_reshaped = torch.reshape(tensor, shape)

    return tensor_reshaped


def reshape_numpy(array: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Reshaped an array to a specified shape.

    :param array: A numpy array.
    :param shape: A shape to be reshaped to.
    """

    array_reshaped = np.reshape(array, shape)

    return array_reshaped


def rotate_numpy(array: np.ndarray, num_rotation: int, axes: tuple) -> np.ndarray:
    """
    Rotate an array 90 degrees a number of times over specified axes.

    :param array: A numpy array.
    :param num_rotation: Number of 90 degree rotations.
    :param axes: The axes to rotate over.
    """

    array_rotated = np.rot90(array, num_rotation, axes)

    return array_rotated
