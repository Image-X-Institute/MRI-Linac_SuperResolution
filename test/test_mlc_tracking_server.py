"""Script used to simulate a waiting MLC tracking socket connection."""

import argparse
import socket
import struct

import numpy as np
import matplotlib.pyplot as plt

HOST = "localhost"
PORT = 31000


def parse_cmd_args() -> dict:
    """
    Function to utilise argparse to extract the user's parameters. Returns
    a dictionary representation of provided arguments.
    """
    parser = argparse.ArgumentParser(
        description="Set-up a test server to receive data from Gadgetron."
    )
    parser.add_argument(
        "--sdim",
        "-s",
        type=int,
        required=True,
        help="The size of the incoming images (e.g., 64 or 256)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        default=False,
        action="store_true",
        help="Plot the first incoming image?",
    )
    args = parser.parse_args()

    return vars(args)


def parse_unpacked_data(unpacked_data: tuple) -> tuple:
    headersize = unpacked_data[0]
    imagesize = unpacked_data[1]
    header = unpacked_data[2:11]
    image = unpacked_data[11:]

    parsed_unpacked_data = (headersize, imagesize, header, image)

    return parsed_unpacked_data


def process_bytes(
    data: bytes, unpacker: struct.Struct, args_dict: dict, counter: int
) -> None:
    unpacked_data = unpacker.unpack(data)
    parsed_unpacked_data = parse_unpacked_data(unpacked_data)

    headersize = parsed_unpacked_data[0]
    imagesize = parsed_unpacked_data[1]
    header = parsed_unpacked_data[2]

    image = np.asarray(parsed_unpacked_data[3]).reshape(
        args_dict["sdim"], args_dict["sdim"]
    )

    print(f"headersize: {headersize}")
    print(f"imagesize: {imagesize}")
    print(f"header: {header}")

    if args_dict["plot"] and counter == 0:
        plt.imshow(image, cmap="gray", vmin=0, vmax=4096)

        plt.show()

    return None


def main():
    args_dict = parse_cmd_args()
    unpacker = struct.Struct(2 * "I" + 7 * "d" + 2 * "i" + args_dict["sdim"] ** 2 * "H")
    print(
        f"Unpacker struct initialised, expecting data of size: {unpacker.size} bytes."
    )
    data = bytes()  # Initialise empty bytes object.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            counter = 0
            print(f"Connected by {addr}.")
            while True:
                _data = conn.recv(4096)  # Take raw data from socket connection.
                if _data:
                    data += _data
                else:
                    pass

                if len(data) == unpacker.size:  # Process once all data is received.
                    print(f"Received image {counter+1}")
                    process_bytes(data, unpacker, args_dict, counter)
                    data = bytes()  # Reinitialise as an empty bytes object.
                    counter += 1

                elif len(data) > unpacker.size:  # Process once all data is received.
                    print(f"Received image {counter+1}")
                    process_bytes(data[: unpacker.size], unpacker, args_dict, counter)
                    data = data[
                        unpacker.size :
                    ]  # Reinitialise as an empty bytes object.
                    counter += 1

                else:
                    pass


if __name__ == "__main__":
    main()
