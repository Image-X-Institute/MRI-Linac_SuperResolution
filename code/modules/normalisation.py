from abc import ABC, abstractmethod
import typing

import torch
import numpy as np


class Normalisation(ABC):
    __slots__ = "unnormalised_data"

    def __init__(
        self, unnormalised_data: typing.Union[torch.Tensor, np.ndarray]
    ) -> None:
        """
        The unnormalised tensor or array is the input to the constructor
        of this base class used in each child class.
        """
        self.unnormalised_data = unnormalised_data

    @abstractmethod
    def apply_normalisation(self):
        pass

    @abstractmethod
    def apply_inverse_normalisation(self):
        pass
