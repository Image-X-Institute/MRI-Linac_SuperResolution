from abc import ABC, abstractmethod
from dataclasses import dataclass

import gadgetron


@dataclass
class Parser(ABC):
    """
    This base class is responsible for parsing basic information
    from the acquisition. DO NOT ALTER THE INFORMATION IN THIS
    BASE CLASS - ALTER IN STRUCTMAKER CLASSES (e.g., multiplying
    by a scale factor for super-resolution).
    """

    acquisition: gadgetron.types.image_array.ImageArray
    connection: gadgetron.external.connection.Connection

    @abstractmethod
    def retrieve_acquisition_data(self) -> dict:
        pass

    @abstractmethod
    def retrieve_connection_data(self) -> dict:
        pass
