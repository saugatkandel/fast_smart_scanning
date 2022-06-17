import abc
from abc import ABC
import numpy as np
import tifffile as tif


class MeasurementInterface(ABC):

    @abc.abstractmethod
    def perform_measurement(self):
        pass


class TransmissionSimulationMeasurementInterface(MeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None):
        assert (image is not None) or (image_path is not None)
        self.image_path = image_path
        if image is not None:
            self.image = image
        else:
            self.image = tif.imread(self.image_path)

    def perform_measurement(self, idxs):
        return self.image[idxs[:, 0], idxs[:, 1]]
