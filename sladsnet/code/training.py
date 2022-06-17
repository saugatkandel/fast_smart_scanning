import copy
from pathlib import Path

import joblib
import numpy as np
import tifffile as tif
from tqdm.notebook import tqdm

from .base import SimulatedSample
from .erd import SladsSklearnModel, SladsModelParams
from .measurement_interface import TransmissionSimulationMeasurementInterface
from .results import Result
from .sampling import run_sampling
from ..input_params import TrainingInputParams, GeneralInputParams, ERDInputParams, SampleParams


class SimulatedSampleParams(SampleParams):
    def __init__(self, image, *args, **kwargs):
        self.image = image
        super().__init__(image.shape, *args, **kwargs)


def generate_database_per_c(sample_params_all, c_value,
                            sampling_type,
                            num_repeats_per_mask=1,
                            measurements_per_initial_mask=1,
                            calculate_full_erd_per_step=True):
    # For the number of mask iterations specified, create new masks and scan them with the specified method

    mask_num_list = []
    params_erd = ERDInputParams(c_value=c_value,
                                static_window=False,
                                calculate_full_erd_per_step=calculate_full_erd_per_step)
    params_gen = GeneralInputParams(num_neighbors=10)
    results_all = Result(c_value=c_value)
    for index in tqdm(range(0, len(sample_params_all)), desc='Samples', leave=False, ascii=True):
        for mask_num in tqdm(range(0, num_repeats_per_mask), desc='Masks', leave=False, ascii=True):
            # Make a copy of the sample
            sample_params = copy.deepcopy(sample_params_all[index])
            sample_params.rng = sample_params_all[index].rng
            measurement_interface = TransmissionSimulationMeasurementInterface(sample_params.image)

            if sampling_type == 'slow_detailed':
                _emulate_experimental_sampling(sample_params, measurement_interface, params_erd, params_gen,
                                               results_all)
            elif sampling_type == 'fast_limited':
                _run_fast_limited_sampling(sample_params, measurement_interface,
                                           params_erd, params_gen, measurements_per_initial_mask, results_all)
    return results_all


def save_database_per_c(results_all: Result,
                        results_dir: str,
                        shuffle: bool = True,
                        training_split: float = 0.9,
                        save_type: str = 'slads-net-reduced'):
    assert save_type in ['slads-net-reduced', 'generic']
    results_dir = Path(results_dir)

    # Separate training and validation data from results
    training_database, validation_database = [], []
    indices_all = np.arange(results_all.size)
    if shuffle:
        np.random.shuffle(indices_all)
    for index in tqdm(range(results_all.size), desc='Set Separation', leave=False, ascii=True):

        results_this = results_all.get_by_index(indices_all[index])
        results_to_write = {}
        if save_type == 'slads-net-reduced':
            results_to_write['poly_features'] = results_this['poly_features']
            erd_this = results_this['erds']
            mask_this = results_this['masks']
            erd_vals = erd_this[~mask_this]
            results_to_write['erds'] = erd_vals
        else:
            results_to_write = results_this

        # Determine if result data should go into training or validation sets
        if index < results_all.size * training_split:
            training_database.append(results_to_write)
        else:
            validation_database.append(results_to_write)

    # Save the complete databases
    with open(results_dir / 'training_database.pkl', 'wb') as f:
        print('training db...')
        joblib.dump(training_database, f)
    with open(results_dir / 'validation_database.pkl', 'wb') as f:
        print('validation db...')
        joblib.dump(validation_database, f)

    return training_database, validation_database


def _emulate_experimental_sampling(sample_params, measurement_interface, params_erd, params_gen, results):
    """Find new measurement position like in the experimental sampling runs"""
    (sample_params.initial_idxs,
     sample_params.initial_mask) = sample_params.generate_initial_mask(sample_params.scan_method)

    sample = SimulatedSample(simulation_type='training',
                             sample_params=sample_params,
                             general_params=params_gen,
                             erd_params=params_erd,
                             measurement_interface=measurement_interface,
                             erd_model=None)
    run_sampling(sample, results=results)


def _run_fast_limited_sampling(sample_params, measurement_interface,
                               params_erd, params_gen, measurements_per_initial_mask,
                               results):
    start_ratio = sample_params.initial_scan_ratio
    stop_ratio = sample_params.stop_ratio

    rand_dist = sample_params.rng.random(measurements_per_initial_mask)
    test_ratios = np.sort((stop_ratio - start_ratio) * rand_dist + start_ratio)

    for ratio in tqdm(test_ratios, leave=False):
        p_this = copy.deepcopy(sample_params)
        p_this.initial_scan_ratio = ratio
        (p_this.initial_idxs,
         p_this.initial_mask) = p_this.generate_initial_mask(p_this.scan_method)

        sample = SimulatedSample(simulation_type='training',
                                 sample_params=p_this,
                                 general_params=params_gen,
                                 erd_params=params_erd,
                                 measurement_interface=measurement_interface,
                                 erd_model=None)
        run_sampling(sample, max_iterations=1, results=results)


def generate_training_databases(train_params: TrainingInputParams, save_type='slads-net-reduced'):
    imgs_path = Path(train_params.input_images_path)
    tif_names = imgs_path.glob('*.tif')

    img_data_all = [tif.imread(f) for f in tif_names]
    rng = np.random.default_rng(train_params.random_seed)
    sim_sample_params = [SimulatedSampleParams(image=img,
                                               initial_scan_ratio=train_params.initial_scan_ratio,
                                               stop_ratio=train_params.stop_ratio,
                                               scan_method=train_params.scan_method,
                                               scan_type=train_params.scan_type,
                                               rng=rng)
                         for img in img_data_all]

    output_dir = Path(train_params.output_dir)
    output_dir.mkdir(exist_ok=True)
    for c_value in tqdm(train_params.test_c_values, leave=True):
        print(f'Testing for c={c_value:4.3g}')
        results_all = generate_database_per_c(sample_params_all=sim_sample_params,
                                              c_value=c_value,
                                              sampling_type=train_params.sampling_type,
                                              measurements_per_initial_mask=train_params.measurements_per_initial_mask,
                                              num_repeats_per_mask=train_params.num_repeats_per_mask,
                                              calculate_full_erd_per_step=train_params.calculate_full_erd_per_step)

        out_dir_this = output_dir / f'c_{c_value}'
        out_dir_this.mkdir(exist_ok=True)
        save_database_per_c(results_all=results_all,
                            results_dir=out_dir_this,
                            training_split=0.9,
                            save_type=save_type)


def _get_features_and_erds_from_db(db_file_path: Path, save_type: str = 'slads-net-reduced'):
    with open(db_file_path, 'rb') as f:
        training_db = joblib.load(f)

    features_all = []
    erds_all = []
    for state in training_db:
        feats_this = state['poly_features']
        features_all.append(feats_this)
        if save_type != 'slads-net-reduced':
            erd_this = state['erds']
            mask_this = state['masks']
            erd_vals = erd_this[~mask_this.astype('bool')]
        else:
            erd_vals = state['erds']
        erds_all = np.concatenate((erds_all, erd_vals))

    features_all = np.concatenate(features_all, axis=0)
    return features_all, erds_all


def fit_erd_model(training_db_path: str,
                  model_type: str = 'slads-net',
                  model_params: SladsModelParams = None,
                  save_path: str = None,
                  save_type: str = 'slads-net-reduced'):
    if model_type != 'slads-net':
        raise NotImplementedError
    training_db_path = Path(training_db_path)
    features_all, erds_all = _get_features_and_erds_from_db(training_db_path, save_type)

    erd_model = SladsSklearnModel(model_params=model_params)
    erd_model.fit(features_all, erds_all)

    if save_path is None:
        save_path = training_db_path.parent / 'erd_model.pkl'
    erd_model.save(save_path)
    return erd_model, save_path


def validate_erd_model_r_squared(validation_db_path: str,
                                 erd_model: SladsSklearnModel = None,
                                 erd_model_path: str = None,
                                 model_type: str = 'slads-net',
                                 save_type: str = 'slads-net-reduced'):
    assert (erd_model is not None) or (erd_model_path is not None)

    if erd_model is None:
        erd_model = SladsSklearnModel(load_path=erd_model_path)

    if model_type != 'slads-net':
        raise NotImplementedError

    features_all, erds_all = _get_features_and_erds_from_db(validation_db_path, save_type)

    if model_type != 'slads-net':
        raise
    score = erd_model.model.score(features_all, erds_all)
    return score
