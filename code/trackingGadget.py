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

from modules.schemas.mlc_tracking import (
    MLCImageArrayParser,
    MLCSocketmaker,
    MLCStructmaker,
)


def main(connection):
    logging.basicConfig(level=logging.DEBUG)

    with MLCSocketmaker() as MLCsm:
        MLCsm.create_socketclient()
        MLCsm.connect_socketclient()

        for acquisition in connection:
            data = acquisition.data

            data_cp = np.copy(data)

            transformation_class = BaseImageArrayTransformations(data_cp)

            image = transformation_class.transform()

            normalisation_class = BaseImageArrayNormalisation(image)

            image = normalisation_class.apply_normalisation()

            parser_class = MLCImageArrayParser(acquisition, connection)

            acquisition_data = parser_class.retrieve_acquisition_data()

            connection_data = parser_class.retrieve_connection_data()

            structmaker_class = MLCStructmaker(
                acquisition_data, connection_data, image, upsample_ratio=1
            )

            packed_struct = structmaker_class.pack_struct()

            logging.info(f"{structmaker_class.header}")

            MLCsm.send_packed_struct(packed_struct)

            image_to_send = ismrmrd.image.Image.from_array(
                image, image_type=ismrmrd.IMTYPE_MAGNITUDE, transpose=True
            )

            connection.send(image_to_send)
