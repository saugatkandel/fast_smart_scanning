import abc
from pathlib import Path
import paramiko
import os

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

class ExperimentMeasurementInterface(MeasurementInterface):
    def __init__(self):#, store_file_scan_points_num: int, num_iterative_idxs: int, 
                    #    num_initial_idxs: int = 2000, 
                    #    is_initialized: bool = False):

        #self._num_initial_idxs = num_initial_idxs
        #self._num_iterative_idxs = num_iterative_idxs
        #self._store_file_scan_points_num = store_file_scan_points_num
        #self._list_idxs = []

        #if not is_initialized:
        #    self._initialized = False
        #    self.current_file_suffix = 0
        #else:
        ##    self._initialized = True
        #    self.current_file_suffix = 1
        #self.current_file_position = 0

        self.new_values = []
        #self._external_measurement_initialized = False
        #self._external_measurement_finalized = False
    """
    def _get_current_fname(self, prefix:str = 'instructions_'):
        return f'{prefix}{self.current_file_suffix:03d}.csv'

    def _check_steps_for_new_file(self):
        new_file_created = False
        if len(self._list_idxs) == self._store_file_scan_points_num and self._initialized:
            self._list_idxs = []
            self.current_file_suffix += 1
            self.current_file_position = 0
            new_file_created = True
        else:
            self.current_file_position += 1
        return new_file_created


    def _write_idx_local(self):
        print("Number of points in file is", len(self._list_idxs))
        fname = self._get_current_fname()
        np.savetxt(fname, self._list_idxs, delimiter=',', fmt='%10d')
        return fname

    def _update_current_list_idxs(self, idxs):
        if not self._initialized:
            if np.shape(idxs)[0] != self._num_initial_idxs:
                raise ValueError(f'Number of idxs for first measurement must match with the "num_initial_idxs"'\
                    f'supplied when creating the measurement interface.')
            
            self._initialized = True
        else:
            if np.shape(idxs)[0] > self._num_iterative_idxs:
                raise ValueError(f'Number of idxs for new measurement must not be greater than "num_iterative_idxs"'\
                    f'supplied when creating the measurement interface.')
        for idx in idxs:
            self._list_idxs.append(idx)

    def initialize_external_measurement(self, idxs):
        new_file_created = self._check_steps_for_new_file()
        self._update_current_list_idxs(idxs)
        out_fname = self._write_idx_local()
        self._external_measurement_initialized = True
        return out_fname, new_file_created
    """

    def finalize_external_measurement(self, values):
        self.new_values = values
        self._external_measurement_finalized = True

    def perform_measurement(self, new_idxs):
        #if (not self._external_measurement_initialized) or (not self._external_measurement_finalized):
        #    raise ValueError
        #if not self._external_measurement_finalized:
        #self._external_measurement_initialized = False
        #self._external_measurement_initialized = False
        return self.new_values
