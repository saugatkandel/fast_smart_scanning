from pathlib import Path

from sladsnet.input_params import ERDInputParams, GeneralInputParams, SampleParams
from sladsnet.code.measurement_interface import TransmissionSimulationMeasurementInterface#ExternalMeasurementInterface
from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.results import Result
from sladsnet.code.base import Sample
from sladsnet.code.sampling import run_sampling
from sladsnet.utils.utils import renormalize
import time


C_VALUE = 2
train_base_path = Path.cwd().parent / 'ResultsAndData/TrainingData/cameraman/'
erd_model = SladsSklearnModel(load_path=train_base_path / f'c_{C_VALUE}/erd_model_relu.pkl')

inner_batch_size = 50
initial_scan_points_num = 2000
initial_scan_ratio = 0.29

stop_ratio = 0.3
store_results_percentage = 1

affected_neighbors_window_min = 5
affected_neighbors_window_max = 15
full_erd_recalculation_frequency = 1

params_erd = ERDInputParams(c_value=C_VALUE,
                            full_erd_recalculation_frequency=full_erd_recalculation_frequency,
                            affected_neighbors_window_min=affected_neighbors_window_min,
                            affected_neighbors_window_max=affected_neighbors_window_max)
params_gen = GeneralInputParams()

sample_params = SampleParams(image_shape=(600, 400),
                             inner_batch_size=inner_batch_size,
                             #initial_scan_points_num=initial_scan_points_num,
                             initial_scan_ratio=initial_scan_ratio,
                             stop_ratio=stop_ratio,
                             random_seed=11)


#interface = ExternalMeasurementInterface('instructions.csv', num_initial_idxs= 2000)
#interface.perform_measurement(sample_params.initial_idxs)
img_path = Path.cwd().parent /'work_directory/analyze_s26_scans/norm_xrm.tif'
image = renormalize(image_path=img_path)
measurement_interface = TransmissionSimulationMeasurementInterface(image=image)

sample = Sample(sample_params=sample_params,
                general_params=params_gen,
                erd_params=params_erd,
                measurement_interface=measurement_interface,
                erd_model=erd_model)
results = Result()

t1 = time.time()
run_sampling(sample, results=results, results_frequency_percentage=store_results_percentage, max_iterations=10,
 disable_progress_bar=False)

t2 = time.time()
print(f"Time required was {t2-t1:4.3g}")