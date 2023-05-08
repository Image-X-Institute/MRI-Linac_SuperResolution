import logging
import socket
import struct
import socket

import numpy as np

import gadgetron

from modules import parser, socketmaker, structmaker


class MLCImageArrayParser(parser.Parser):
    """
    Child class for extract relevant meta data from the acquisition/connection that needs
    to be sent to the Australian MRI-linac MLC tracking software. From the acquisition, this
    includes the slice position, and width/height of the acquisition. From the connection, this
    is the voxel (in 2D, pixel) size. These are both returned as dictionaries for use with the
    structmaker class to be processed in a way appropriate for the software.
    """

    def retrieve_acquisition_data(self) -> tuple:
        """
        Retrieves available data from the acquisition object.

        :param acquisition: the acquisition provided by the Gadgetron
        connection object.
        """
        SlicePositionSagittal = self.acquisition.acq_headers[0][0][0][0][0].position[0]
        SlicePositionCoronal = self.acquisition.acq_headers[0][0][0][0][0].position[1]
        SlicePositionTransverse = self.acquisition.acq_headers[0][0][0][0][0].position[
            2
        ]
        Width = self.acquisition.data.shape[0]
        Height = self.acquisition.data.shape[1]

        acquisition_data_dict = {
            "SlicePositionSagittal": SlicePositionSagittal,
            "SlicePositionCoronal": SlicePositionCoronal,
            "SlicePositionTransverse": SlicePositionTransverse,
            "Width": Width,
            "Height": Height,
        }

        return acquisition_data_dict

    def retrieve_connection_data(self) -> tuple:
        """
        Retrieves available data from the connection object.

        :param connection: the connection object provied by Gadgetron.
        """

        FOVX = self.connection.header.encoding[0].encodedSpace.fieldOfView_mm.x
        FOVY = self.connection.header.encoding[0].encodedSpace.fieldOfView_mm.y
        FOVZ = self.connection.header.encoding[0].encodedSpace.fieldOfView_mm.z

        connection_data_dict = {"FOVX": FOVX, "FOVY": FOVY, "FOVZ": FOVZ}

        return connection_data_dict


class MLCStructmaker(structmaker.Structmaker):
    HEADERSIZE = 64

    def __init__(
        self,
        acquisition_data: dict,
        connection_data: dict,
        image_data: np.ndarray,
        upsample_ratio: int,
    ) -> None:
        self.upsample_ratio = upsample_ratio

        self.acquisition_data = acquisition_data
        self.process_acquisition_data()

        self.connection_data = connection_data
        self.process_connection_data()

        self.image_data = image_data
        self.process_image_data()

        self.prepare_header()

        self.generate_struct()

    def process_acquisition_data(self) -> None:
        """
        Converts the acquisition data into the required forms.
        """

        for keyname in self.acquisition_data:
            if keyname in ["Width", "Height"]:
                self.acquisition_data[keyname] = int(
                    self.acquisition_data[keyname] * self.upsample_ratio
                )
            else:
                self.acquisition_data[keyname] = float(self.acquisition_data[keyname])
        return None

    def process_connection_data(self) -> None:
        """
        Converts the connection data into the required forms.
        """
        self.connection_data["VoxelSizeX"] = None
        self.connection_data["VoxelSizeY"] = None
        self.connection_data["VoxelSizeZ"] = None

        for keyname in self.connection_data:
            if keyname in [
                "FOVX"
            ]:  # Dividing FOV by width and 2 (to account for 2x RO oversampling).
                self.connection_data["VoxelSizeX"] = float(
                    self.connection_data[keyname] / (2 * self.acquisition_data["Width"])
                )
            elif keyname in ["FOVY"]:  # Dividing FOV by height.
                self.connection_data["VoxelSizeY"] = float(
                    self.connection_data[keyname] / (self.acquisition_data["Height"])
                )
            elif keyname in [
                "FOVZ"
            ]:  # Single slice data currently, so the slice thickness (i.e. VoxelSizeZ) is the FOVZ.
                self.connection_data["VoxelSizeZ"] = float(
                    self.connection_data[keyname]
                )
            else:
                self.connection_data[keyname] = float(self.connection_data[keyname])
        return None

    def process_image_data(self) -> None:
        self.image_data = self.image_data.ravel()
        self.image_data = np.absolute(self.image_data)
        self.image_data = self.image_data.astype(np.int16)
        self.image_size = len(self.image_data)
        return None

    def prepare_header(self) -> None:
        """
        Generates a prepared header in format expected by MLC tracking software.
        The prepared header needs to be a tuple in the following format and order.
        (SlicePositionSagittal,
         SlicePositionCoronal,
         SlicePositionTransverse,
         SliceThickness(VoxelSizeZ),
         SpacingBetweenSlices(FOVZ - num_slices*VoxelSizeZ),
         PixelSizeX(VoxelSizeX),
         PixelSizeY(VoxelSizeY),
         Width,
         Height)
        """
        self.header = (
            self.acquisition_data["SlicePositionSagittal"],
            self.acquisition_data["SlicePositionCoronal"],
            self.acquisition_data["SlicePositionTransverse"],
            self.connection_data["VoxelSizeZ"],
            0.0,  # Hardcoded here as current MLC tracking is 2D only
            self.connection_data["VoxelSizeX"],
            self.connection_data["VoxelSizeY"],
            self.acquisition_data["Width"],
            self.acquisition_data["Height"],
        )
        return None

    def generate_struct(self) -> None:
        """
        Generates a appropriately sized struct.
        """
        self._struct = struct.Struct(
            2 * "I" + 7 * "d" + 2 * "i" + self.image_size * "H"
        )
        return None

    def pack_struct(self) -> bytes:
        """
        Packs the struct with the data to be sent over socket connection.
        """
        self.packed_struct = self._struct.pack(
            self.HEADERSIZE, 2 * self.image_size, *self.header, *self.image_data
        )

        return self.packed_struct


class MLCSocketmaker(socketmaker.Socketmaker):
    HOSTNAME = "localhost"
    PORT = 31000

    def __init__(self) -> None:
        pass

    def create_socketclient(self) -> None:
        logging.info(f"Creating socket client")
        self.socketclient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return None

    def connect_socketclient(self) -> None:
        logging.info(f"Connecting to address: {self.HOSTNAME} on port: {self.PORT}")
        self.socketclient.connect((self.HOSTNAME, self.PORT))
        return None

    def send_packed_struct(self, packed_struct: struct.Struct) -> None:
        self.socketclient.sendall(packed_struct)
        return None

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # Only print error information if an error came through (ignore if no error).
        if not [x for x in (exc_type, exc_value, exc_traceback) if x is None]:
            logging.error(f"Error type: {exc_type}")
            logging.error(f"Error value: {exc_value}")
            logging.error(f"Error traceback: {exc_traceback}")

        self.socketclient.close()  # Close socketclient.
        logging.info(f"Closing socket client")
        return None
