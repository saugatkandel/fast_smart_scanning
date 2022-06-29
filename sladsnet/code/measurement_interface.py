import abc
import numpy as np


class MeasurementInterface(abc.ABC):

    @abc.abstractmethod
    def perform_measurement(self, idxs):
        pass


class TransmissionSimulationMeasurementInterface(MeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None):
        import tifffile as tif
        assert (image is not None) or (image_path is not None)
        self.image_path = image_path
        if image is not None:
            self.image = image
        else:
            self.image = tif.imread(self.image_path)

    def perform_measurement(self, idxs):
        return self.image[idxs[:, 0], idxs[:, 1]]


class ExternalMeasurementInterface(MeasurementInterface):
    def __init__(self):
        """This is all currently handled within the experiment script."""
        self.new_values = []
        self._external_measurement_finalized = False

    def finalize_external_measurement(self, values):
        self.new_values = values
        self._external_measurement_finalized = True

    def perform_measurement(self, new_idxs):
        return self.new_values
