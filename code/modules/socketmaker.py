from abc import ABC, abstractmethod
import struct

class Socketmaker(ABC):
    """
    Abstract base class for managing the socket connection to external
    programs (e.g., MLC tracking software). Intended to work in a context
    manager (see test below) to ensure the closing of the socket connection
    upon completion, or error.
    """

    @abstractmethod
    def __init__(self, hostname: str, port: int, *args) -> None:
        pass

    @abstractmethod
    def create_socketclient(self) -> None:
        pass

    @abstractmethod
    def send_packed_struct(self, packed_struct: struct.Struct) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> None:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass
