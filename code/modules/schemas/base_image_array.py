import numpy as np

from modules import (
    normalisation,
    transformations,
    transformation_utils,
)


class BaseImageArrayNormalisation(normalisation.Normalisation):
    def apply_normalisation(self) -> np.ndarray:
        """
        Apply the normalisation and store the relevant values for inverse
        normalisation if applicable.

        :param unnormalised_data: The data to be normalised.
        """
        normalised_array = self.unnormalised_data

        return normalised_array

    def apply_inverse_normalisation(self) -> None:
        pass


class BaseImageArrayTransformations(transformations.Transformations):
    def transform(self) -> np.ndarray:
        """
        The transformations required to process the data in a form compatible
        with the struct and the rest of Gadgetron are stored here.
        """
        transformed_data = transformation_utils.reshape_numpy(
            self.gadgetron_array,
            (
                self.gadgetron_array.shape[0],
                self.gadgetron_array.shape[1],
                self.gadgetron_array.shape[2],
                self.gadgetron_array.shape[3],
            ),
        )

        transformed_data = transformation_utils.rotate_numpy(
            transformed_data, num_rotation=2, axes=(0, 1)
        )

        return transformed_data
