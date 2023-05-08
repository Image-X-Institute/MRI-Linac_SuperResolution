import numpy as np
from abc import ABC, abstractmethod


class Transformations(ABC):
    """
    Parent class for transformations, abstract methods are
    defined here.
    """

    __slots__ = "gadgetron_array"

    def __init__(self, gadgetron_array: np.ndarray) -> None:
        """
        Constructor calls the Numpy array from Gadgetron to be
        passed into pretransform.

        :param gadgetron_array: The numpy array from Gadgetron.
        """

        self.gadgetron_array = gadgetron_array

    @abstractmethod
    def transform(self):
        pass
