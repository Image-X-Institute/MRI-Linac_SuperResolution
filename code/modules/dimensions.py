"""
In this module, the input and output dimensions for each neural network
are stored. Simply make a dataclass and store the dimensional data.

input_dimensions: the dimensions expected for the neural network application.
output_dimensions: the dimensions expected to be sent back to Gadgetron.
"""

from dataclasses import dataclass


@dataclass
class Dimensions:
    """
    Generic dimensions class.
    """

    __slots__ = "input_dimensions", "output_dimensions"
    input_dimensions: tuple
    output_dimensions: tuple


def test():
    class TestDimensions(Dimensions):
        input_dimensions: tuple = (1, 2, 3, 4)
        output_dimensions: tuple = (5, 6, 7, 8)

    assert TestDimensions.input_dimensions == (1, 2, 3, 4)
    assert TestDimensions.output_dimensions == (5, 6, 7, 8)


if __name__ == "__main__":
    test()
