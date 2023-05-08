import time
import logging

import torch
import numpy as np

import ismrmrd
import gadgetron

from modules.schemas.base_image_array import (
    BaseImageArrayNormalisation,
    BaseImageArrayTransformations,
)


def main(connection):
    logging.basicConfig(level=logging.DEBUG)

    for acquisition in connection:
        data = acquisition.data

        data_cp = np.copy(data)

        transformation_class = BaseImageArrayTransformations(data_cp)

        image = transformation_class.transform()

        normalisation_class = BaseImageArrayNormalisation(image)

        image = normalisation_class.apply_normalisation()

        image_to_send = ismrmrd.image.Image.from_array(
            image, image_type=ismrmrd.IMTYPE_MAGNITUDE, transpose=True
        )

        connection.send(image_to_send)
