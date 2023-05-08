import torch
from abc import ABC, abstractmethod


class Posttransformations(ABC):
    """
    Parent class for pretransformations, abstract methods are
    defined here.
    """

    __slots__ = "inferred_tensor"

    def __init__(self, inferred_tensor: torch.Tensor) -> None:
        """
        Constructor calls inferred tensor after neural network inference.

        :param inferred_tensor: The inferred tensor.
        :param device: device to run inference on.
        """
        self.inferred_tensor = inferred_tensor

    @abstractmethod
    def posttransform(self):
        pass
