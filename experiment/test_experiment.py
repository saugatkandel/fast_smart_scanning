from sladsnet.code.measurement_interface import ExperimentMeasurementInterface
from sladsnet.code.erd import SladsSklearnModel

import numpy as np
from pathlib import Path

from sladsnet.input_params import ERDInputParams, GeneralInputParams, SampleParams

#from sladsnet.code.results import Result
#from sladsnet.code.base import Sample
#from sladsnet.code.sampling import run_sampling
#from sladsnet.code.utils import renormalize


C_VALUE = 2
train_base_path = Path.cwd().parent / 'ResultsAndData/TrainingData/cameraman/'
erd_model = SladsSklearnModel(load_path=train_base_path / f'c_{C_VALUE}/erd_model_relu.pkl')

inner_batch_size = 50
initial_scan_points_num = 2000

stop_ratio = 0.3
store_results_percentage = 1

affected_neighbors_window_min = 5
affected_neighbors_window_max = 15
full_erd_recalculation_frequency = 2

params_erd = ERDInputParams(c_value=C_VALUE,
                            full_erd_recalculation_frequency=full_erd_recalculation_frequency,
                            affected_neighbors_window_min=affected_neighbors_window_min,
                            affected_neighbors_window_max=affected_neighbors_window_max)
params_gen = GeneralInputParams()

sample_params = SampleParams(image_shape=(600, 400),
                             inner_batch_size=inner_batch_size,
                             initial_scan_points_num=initial_scan_points_num,
                             initial_scan_ratio=None,
                             stop_ratio=stop_ratio,
                             random_seed=11)


interface = ExperimentMeasurementInterface('instructions.csv', num_initial_idxs= 2000)
interface.perform_measurement(sample_params.initial_idxs)

print("Sucess")