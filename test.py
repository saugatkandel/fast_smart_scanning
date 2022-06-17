import sladsnet.code.training as training
from sladsnet.input_params import TrainingInputParams
from sladsnet.code.training import SimulatedSampleParams
from sladsnet.code.erd import SladsModelParams
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import joblib
import dataclasses as dt
from tqdm.notebook import tqdm
import tifffile as tif
from sladsnet.input_params import ERDInputParams, GeneralInputParams
from sladsnet.code.measurement_interface import TransmissionSimulationMeasurementInterface
from sladsnet.code.results import Result
from sladsnet.code.base import SimulatedSample
from sladsnet.code.sampling import run_sampling
from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.neighbors import find_neighbors



train_params = TrainingInputParams(num_repeats_per_mask=2, measurements_per_initial_mask=50, 
                                   test_c_values=[16])
                                   #test_c_values=[2, 4, 8, 16, 32, 64])
    

base_path = Path.cwd() / Path('ResultsAndData/TrainingData/TrainingDB_1')

params_erd = ERDInputParams(c_value=16, static_window=False, calculate_full_erd_per_step=False,
                           affected_neighbors_window_min=10, affected_neighbors_window_max=20)
params_gen = GeneralInputParams(num_neighbors=10)

imgs_path = Path(train_params.input_images_path)
tif_names = imgs_path.glob('*.tif')

img_data_all = [tif.imread(f) for f in tif_names]
rng = np.random.default_rng()
sample_params = SimulatedSampleParams(image=img_data_all[0], points_to_scan=1,
                                     initial_scan_ratio=0.01,
                                     stop_ratio=0.2,
                                     scan_method='pointwise',
                                     scan_type='transmission',
                                     rng=rng)

results = Result()
measurement_interface = TransmissionSimulationMeasurementInterface(sample_params.image.copy())erd_model = SladsSklearnModel(load_path=base_path / Path(f'c_16/erd_model_relu.pkl'))
sample = SimulatedSample(simulation_type='c_opt',
                                  sample_params=sample_params,
                                  general_params=params_gen,
                                  erd_params=params_erd,
                                  measurement_interface=measurement_interface,
                                  erd_model=erd_model)