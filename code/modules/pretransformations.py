import torch
import numpy as np
from abc import ABC, abstractmethod


class Pretransformations(ABC):
    """
    Parent class for pretransformations, abstract methods are
    defined here.
    """

    __slots__ = "gadgetron_array", "device"

    def __init__(self, gadgetron_array: np.ndarray, device: torch.device) -> None:
        """
        Constructor calls the Numpy array from Gadgetron to be
        passed into pretransform.

        :param gadgetron_array: The numpy array from Gadgetron.
        :param device: device to run inference on.
        """

        self.gadgetron_array = gadgetron_array
        self.device = device

    @abstractmethod
    def pretransform(self):
        pass
