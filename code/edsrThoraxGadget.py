import time
import logging
import os

import numpy as np
import torch

import ismrmrd
import gadgetron

from modules.schemas.edsr_sr import (
    EdsrModel,
    EdsrPretransformations,
    EdsrPosttransformations,
)


def main(connection):
    logging.basicConfig(level=logging.DEBUG)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.debug(f"Device used for inference: {device}")

    model = EdsrModel(device, "2022-09-15_08-58-43_edsr_thorax_nonoise.pt")

    for acquisition in connection:
        data = acquisition.data

        data_cp = np.copy(data)

        pretransformation_class = EdsrPretransformations(data_cp, device)

        image = pretransformation_class.pretransform()

        image_inferred = model.perform_inference(image)

        posttransformation_class = EdsrPosttransformations(image_inferred)

        image_inferred = posttransformation_class.posttransform()

        image_to_send = ismrmrd.image.Image.from_array(
            image_inferred, image_type=ismrmrd.IMTYPE_MAGNITUDE, transpose=True
        )

        connection.send(image_to_send)
