#
# Authors 06/29/22:
# Saugat Kandel: for the SLADS parts
# Tao Zhou: for the EPICS and the Tkinter parts
# Mathew Cherukara: General consulting
#
# Uses tkinter and uses epics triggers generated in ives (Sector26 computer) to run the measurement.
#
import fnmatch
import logging
import shutil
from pathlib import Path

import numpy as np
from paramiko import SFTPClient

from sladsnet.code.base import ExperimentalSample
from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.measurement_interface import ExternalMeasurementInterface
from sladsnet.input_params import ERDInputParams, GeneralInputParams, SampleParams


def load_idxs_and_intensities(data_dir: str):
    """Directory containing the npz with inital positions and intensities.
     These are generated from MDA files by Tao at sector 26."""
    initial_data_path = Path(data_dir)
    npz_files = list(initial_data_path.glob('init_*.npz'))

    num_files = len(npz_files)
    initial_idxs = np.empty((0, 2), dtype='int')
    initial_intensities = np.empty(0, dtype='int')
    for fnpz in npz_files:
        fname = str(fnpz)
        logging.info('Loading initial data from %s' % fname)
        initial_data = np.load(fname)
        # Need to shift the idxs so that it starts from 0.
        xpoints = initial_data['x'][:-1]
        ypoints = initial_data['y'][:-1]
        idxs_this = np.array((ypoints, xpoints), dtype='int').T
        intensities = initial_data['I'][1:]

        initial_idxs = np.concatenate((initial_idxs, idxs_this), axis=0)
        initial_intensities = np.concatenate((initial_intensities, intensities))

    return num_files, initial_idxs, initial_intensities


def clean_data_directory(data_dir: str):
    """This is dangerous. Use with care."""
    dpath = Path(data_dir)
    if dpath.exists():
        logging.warning('Removing the files currently present in %s.' % data_dir)
        shutil.rmtree(data_dir)
    dpath.mkdir()


def get_wildcard_files_remote(sftp: SFTPClient, remote_dir: str, search: str):
    matching_filenames = []
    for filename in sftp.listdir(remote_dir):
        if fnmatch.fnmatch(filename, search):
            matching_filenames.append(filename)
    return matching_filenames


def get_init_npzs_from_remote(sftp: SFTPClient, remote_dir: str, data_dir: str):
    init_npzs = get_wildcard_files_remote(sftp, remote_dir, 'init*.npz')
    for f in init_npzs:
        sftp.get(f, data_dir)


def create_experiment_sample(numx: int, numy: int,
                             initial_idxs: list,
                             inner_batch_size: int = 100,
                             stop_ratio: float = 0.35,
                             c_value: float = 2.0,
                             full_erd_recalculation_frequency: int = 1,
                             affected_neighbors_window_min: int = 5,
                             affected_neighbors_window_max: int = 15,
                             erd_model_file_path: str = None):
    if erd_model_file_path is None:
        erd_model_file_path = Path(__file__).parent.parent / 'ResultsAndData/TrainingData/cameraman/' \
                              / f'c_{c_value}/erd_model_relu.pkl'
    erd_model = SladsSklearnModel(load_path=erd_model_file_path)

    params_erd = ERDInputParams(c_value=c_value,
                                full_erd_recalculation_frequency=full_erd_recalculation_frequency,
                                affected_neighbors_window_min=affected_neighbors_window_min,
                                affected_neighbors_window_max=affected_neighbors_window_max)
    params_gen = GeneralInputParams()

    sample_params = SampleParams(image_shape=(numy, numx),
                                 inner_batch_size=inner_batch_size,
                                 initial_idxs=initial_idxs,
                                 stop_ratio=stop_ratio,
                                 random_seed=11)

    measurement_interface = ExternalMeasurementInterface()

    sample = ExperimentalSample(sample_params=sample_params,
                                general_params=params_gen,
                                erd_params=params_erd,
                                measurement_interface=measurement_interface,
                                erd_model=erd_model)
    return sample
