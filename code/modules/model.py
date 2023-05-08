import time
import logging
from abc import abstractmethod

import torch

from modules import model_utils


class Model:
    """
    This base class is responsible for loading the model.
    """

    DOCKER = True  # Set this variable to True if using Docker container.
    DOCKERPATH = "/opt/conda/envs/gadgetron/share/gadgetron/python"
    MRLPATH = "parameters/"

    __slots__ = "device", "model_name"

    def __init__(self, device: torch.device, model_name: str) -> None:
        """
        :param device: device to run inference on.
        :param model_name: name of the JIT compiled model.
        """
        self.device = device
        self.model_name = model_name
        self.load_model()
        self.initialise_gpu(self.model)

    def load_model(self) -> None:
        """
        Load the JIT compiled model (containing the trained parameters) onto the
        GPU. Will display logging information confirming device loaded and time
        taken to load.
        """
        start_model_load = time.time()

        path_to_parameters = model_utils.configure_path_to_parameters(
            Model.DOCKER, Model.DOCKERPATH, Model.MRLPATH, self.model_name
        )

        self.model = torch.jit.load(path_to_parameters, map_location=self.device).eval()

        logging.debug(f"Model loaded! Time taken: {time.time()-start_model_load:.3f}")

        return None

    @abstractmethod
    def initialise_gpu(self):
        pass

    @abstractmethod
    def perform_inference(self):
        pass
