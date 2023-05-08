from dataclasses import dataclass
import logging
import time

import numpy as np
import torch

from modules import (
    dimensions,
    model,
    pretransformations,
    posttransformations,
    transformation_utils,
)


@dataclass
class EdsrDimensions(dimensions.Dimensions):
    """
    Edsr dimensions class. Store data here.
    """

    input_dimensions: tuple = (1, 1, 64, 64)
    output_dimensions: tuple = (256, 256, 1, 1)


class EdsrModel(model.Model):
    """
    Child class of Model. Methods specific to application are defined here.
    """

    def initialise_gpu(self, model: torch.nn.Module) -> None:
        """
        Pass through some random input to warm the GPU up for real-time
        streaming.

        :param model: the loaded in model
        """
        logging.info("Warming up GPU.")
        model(
            torch.rand(EdsrDimensions.input_dimensions).to(
                self.device, dtype=torch.float
            )
        )

        return None

    def perform_inference(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on a pretransformed, normalised input tensor.
        """
        start_inference = time.time()
        image_superresolution = self.model(input_data)
        logging.info(f"Edsr inference time: {time.time()-start_inference:.5f}")

        return image_superresolution


class EdsrPretransformations(pretransformations.Pretransformations):
    def pretransform(self) -> torch.Tensor:
        """
        This method is responsible for defining the specific pretransformation
        steps for the application. The helper functions from transformation_utils
        can be utilised here depending on use-case. Specific, non-transferrable
        transformations are encouraged to be defined in this child class.
        """
        tensor_data = transformation_utils.convert_numpy2tensor(
            self.gadgetron_array, self.device  # Shape (sdim, sdim, 1, 1, 1, 1, 1)
        )

        tensor_data = transformation_utils.squeeze(tensor_data)

        tensor_data = transformation_utils.get_magnitude_tensor(tensor_data)

        tensor_data = transformation_utils.rotate(tensor_data, num_rotation=2)

        tensor_data = transformation_utils.unsqueeze(tensor_data)

        tensor_data = transformation_utils.unsqueeze(tensor_data)

        return tensor_data


class EdsrPosttransformations(posttransformations.Posttransformations):
    def posttransform(self) -> np.ndarray:
        """
        This method is responsible for defining the specific posttransformation
        steps for the application. The helper functions from transformation_utils
        can be utilised here depending on use-case. Specific, non-transferrable
        transformations are encouraged to be defined in this child class.
        """
        tensor_data = transformation_utils.reshape(
            self.inferred_tensor, EdsrDimensions.output_dimensions
        )

        array_data = transformation_utils.convert_tensor2numpy(tensor_data)

        return array_data
