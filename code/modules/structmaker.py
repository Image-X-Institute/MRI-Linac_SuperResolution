from abc import ABC, abstractmethod


class Structmaker(ABC):
    """
    This base class is responsible for preparing the struct
    to be sent through a socket (e.g., to the MLC tracking
    software).
    """

    @abstractmethod
    def __init__(self, acquisition_data: dict, connection_data: dict, *args) -> None:
        self.acquisition_data = acquisition_data
        self.connection_data = connection_data

    @abstractmethod
    def process_acquisition_data(self) -> None:
        pass

    @abstractmethod
    def process_connection_data(self) -> None:
        pass

    @abstractmethod
    def generate_struct(self) -> None:
        pass

    @abstractmethod
    def pack_struct(self) -> bytes:
        pass
