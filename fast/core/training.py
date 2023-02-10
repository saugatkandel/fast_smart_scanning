# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Software Name:    Fast Autonomous Scanning Toolkit (FAST)               #
# By: Argonne National Laboratory                                         #
# OPEN SOURCE LICENSE                                                     #
#                                                                         #
# Redistribution and use in source and binary forms, with or without      #
# modification, are permitted provided that the following conditions      #
# are met:                                                                #
#                                                                         #
# 1. Redistributions of source code must retain the above copyright       #
#    notice, this list of conditions and the following disclaimer.        #
#                                                                         #
# 2. Redistributions in binary form must reproduce the above copyright    #
#    notice, this list of conditions and the following disclaimer in      #
#    the documentation and/or other materials provided with the           #
#    distribution.                                                        #
#                                                                         #
# 3. Neither the name of the copyright holder nor the names of its        #
#    contributors may be used to endorse or promote products derived from #
#    this software without specific prior written permission.             #
#                                                                         #
# *********************************************************************** #
#                                                                         #
# DISCLAIMER                                                              #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE          #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,    #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS   #
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED      #
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF   #
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY            #
# OF SUCH DAMAGE.                                                         #
# *********************************************************************** #
import copy
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import tifffile as tif
from tqdm.notebook import tqdm

from .. import input_params as inp


from ..utils.renormalize import renormalize
from .erd import SladsModelParams, SladsSklearnModel
from .measurement_interface import TransmissionSimulationMeasurementInterface
from .results import Result
from .sampling import run_sampling
from .simulation_base import SimulatedSample


def generate_database_per_c(
    sample_params_all,
    c_value,
    sampling_type,
    num_repeats_per_mask=1,
    measurements_per_initial_mask=1,
    calculate_full_erd_per_step=True,
):
    # For the number of mask iterations specified, create new masks and scan them with the specified method

    mask_num_list = []
    params_erd = inp.ERDInputParams(
        c_value=c_value,
        static_window=False,
        calculate_full_erd_per_step=calculate_full_erd_per_step,
    )
    params_gen = inp.GeneralInputParams(num_neighbors=10)
    results_all = Result(c_value=c_value)
    for index in tqdm(
        range(0, len(sample_params_all)),
        desc=f"Testing for c={c_value:4.3g}; Samples",
        leave=False,
        ascii=True,
    ):
        for mask_num in tqdm(range(0, num_repeats_per_mask), desc="Masks", leave=False, ascii=True):
            # Make a copy of the sample
            sample_params = copy.deepcopy(sample_params_all[index])
            sample_params.rng = sample_params_all[index].rng
            measurement_interface = TransmissionSimulationMeasurementInterface(sample_params.image)

            if sampling_type == "slow_detailed":
                _emulate_experimental_sampling(
                    sample_params,
                    measurement_interface,
                    params_erd,
                    params_gen,
                    results_all,
                )
            elif sampling_type == "fast_limited":
                _run_fast_limited_sampling(
                    sample_params,
                    measurement_interface,
                    params_erd,
                    params_gen,
                    measurements_per_initial_mask,
                    results_all,
                )
    return results_all


def save_database_per_c(results_all: Result, results_dir: str, save_type: str = "slads-net-reduced"):
    assert save_type in inp.SUPPORTED_SAVE_DATABASE_TYPES
    results_dir = Path(results_dir)

    training_database = []
    for index in range(results_all.size):
        results_this = results_all.get_by_index(index)
        results_to_write = {}
        if save_type == "slads-net-reduced":
            results_to_write["poly_features"] = results_this["poly_features"]
            erd_this = results_this["erds"]
            mask_this = results_this["masks"]
            erd_vals = erd_this[~mask_this]
            results_to_write["erds"] = erd_vals
        else:
            results_to_write = results_this

        training_database.append(results_to_write)

    # Save the complete databases
    with open(results_dir / "training_database.pkl", "wb") as f:
        joblib.dump(training_database, f)
    return training_database


def _emulate_experimental_sampling(sample_params, measurement_interface, params_erd, params_gen, results):
    """Find new measurement position like in the experimental sampling runs"""
    (
        sample_params.initial_idxs,
        sample_params.initial_mask,
    ) = sample_params.generate_initial_mask(sample_params.scan_method)

    sample = SimulatedSample(
        sample_params=sample_params,
        general_params=params_gen,
        erd_params=params_erd,
        measurement_interface=measurement_interface,
        erd_model=None,
    )
    run_sampling(sample, results=results)


def _run_fast_limited_sampling(
    sample_params,
    measurement_interface,
    params_erd,
    params_gen,
    measurements_per_initial_mask,
    results,
):
    start_ratio = sample_params.initial_scan_ratio
    stop_ratio = sample_params.stop_ratio

    rand_dist = sample_params.rng.random(measurements_per_initial_mask)
    test_ratios = np.sort((stop_ratio - start_ratio) * rand_dist + start_ratio)

    for ratio in tqdm(test_ratios, leave=False, desc="Iterating through test sampling ratios."):
        p_this = copy.deepcopy(sample_params)
        p_this.initial_scan_ratio = ratio
        (p_this.initial_idxs, p_this.initial_mask) = p_this.generate_initial_mask(p_this.scan_method)

        sample = SimulatedSample(
            sample_params=p_this,
            general_params=params_gen,
            erd_params=params_erd,
            measurement_interface=measurement_interface,
            erd_model=None,
        )
        run_sampling(sample, max_iterations=1, results=results)


def generate_training_databases(train_params: inp.TrainingInputParams):
    imgs_path = Path(train_params.input_images_path)
    tif_names = imgs_path.glob("*.tif")

    img_data_all = [renormalize(tif.imread(f)) for f in tif_names]
    rng = np.random.default_rng(train_params.random_seed)
    sim_sample_params = [
        inp.SimulatedSampleParams(
            image=img,
            simulation_type="training",
            initial_scan_ratio=train_params.initial_scan_ratio,
            stop_ratio=train_params.stop_ratio,
            scan_method=train_params.scan_method,
            scan_type=train_params.scan_type,
            initial_mask_type=train_params.initial_mask_type,
            rng=rng,
        )
        for img in img_data_all
    ]

    output_dir = Path(train_params.output_dir)
    output_dir.mkdir(exist_ok=True)
    pbar = tqdm(train_params.test_c_values, leave=True)
    results_all = []
    for c_value in pbar:
        pbar.set_postfix({f"Generating database for c": c_value})
        results_c = generate_database_per_c(
            sample_params_all=sim_sample_params,
            c_value=c_value,
            sampling_type=train_params.sampling_type,
            measurements_per_initial_mask=train_params.measurements_per_initial_mask,
            num_repeats_per_mask=train_params.num_repeats_per_mask,
            calculate_full_erd_per_step=train_params.calculate_full_erd_per_step,
        )

        out_dir_this = output_dir / f"c_{c_value}"
        out_dir_this.mkdir(exist_ok=True)
        save_database_per_c(
            results_all=results_c,
            results_dir=out_dir_this,
            save_type=train_params.save_type,
        )
        results_all.append(results_c)
    return results_all


def get_features_and_erds_from_db(db_file_path: Path, save_type: str = "slads-net-reduced"):
    with open(db_file_path, "rb") as f:
        training_db = joblib.load(f)

    features_all = []
    erds_all = []
    for state in training_db:
        feats_this = state["poly_features"]
        features_all.append(feats_this)
        if save_type != "slads-net-reduced":
            erd_this = state["erds"]
            mask_this = state["masks"]
            erd_vals = erd_this[~mask_this.astype("bool")]
        else:
            erd_vals = state["erds"]
        erds_all = np.concatenate((erds_all, erd_vals))

    features_all = np.concatenate(features_all, axis=0)
    return features_all, erds_all


def training_validation_split(
    features_all: npt.NDArray,
    erds_all: npt.NDArray,
    training_split: float = 0.8,
    shuffle: bool = True,
    random_seed: int = None,
):
    rng = np.random.default_rng(random_seed)
    indices_all = np.arange(features_all.shape[0])
    if shuffle:
        rng.shuffle(indices_all)

    features_shuffled = features_all[indices_all]
    erds_shuffled = erds_all[indices_all]

    train_num = np.ceil(indices_all.size * training_split).astype("int")
    train_features = features_shuffled[:train_num]
    train_erds = erds_shuffled[:train_num]

    validation_features = features_shuffled[train_num:]
    validation_erds = erds_shuffled[train_num:]
    return train_features, train_erds, validation_features, validation_erds


def fit_erd_model(
    train_features_erds: tuple[npt.NDArray, npt.NDArray] = None,
    full_features_erds_db_path: str = None,
    training_split: float = 0.8,
    random_seed: int = 111,
    model_type: str = "slads-net",
    model_params: SladsModelParams = None,
    save_path: str = None,
    save_type: str = "slads-net-reduced",
):
    if train_features_erds is not None:
        print(
            "Ignoring the features db path and train/val split ratio."
            f"Directly loading the training features and erds instead."
        )
        train_features, train_erds = train_features_erds
    else:
        assert (full_features_erds_db_path is not None) and (0 < training_split < 1)
        print(
            f"Loading the full database and splitting train/val data."
            f"Ensure that the same random seed is used for both this training function and the validation function."
        )
        full_features_erds_db_path = Path(full_features_erds_db_path)
        features_all, erds_all = get_features_and_erds_from_db(full_features_erds_db_path, save_type)
        [
            train_features,
            train_erds,
            validation_features,
            validation_erds,
        ] = training_validation_split(features_all, erds_all, training_split, random_seed=random_seed)

    if model_type != "slads-net":
        raise NotImplementedError
    model_params = model_params if model_params is not None else SladsModelParams()
    erd_model = SladsSklearnModel(model_params=model_params)
    erd_model.fit(train_features, train_erds)

    if save_path is None:
        save_path = full_features_erds_db_path.parent / "erd_model.pkl"
    erd_model.save(save_path)
    return erd_model, save_path


def validate_erd_model_r_squared(
    validate_features_erds: tuple[npt.NDArray, npt.NDArray] = None,
    full_features_erds_db_path: str = None,
    training_split: float = 0.8,
    erd_model: SladsSklearnModel = None,
    erd_model_path: str = None,
    model_type: str = "slads-net",
    save_type: str = "slads-net-reduced",
    random_seed: int = 111,
):
    assert (erd_model is not None) or (erd_model_path is not None)

    if erd_model is None:
        erd_model = SladsSklearnModel(load_path=erd_model_path)

    if model_type != "slads-net":
        raise NotImplementedError

    if validate_features_erds is not None:
        print(
            "Ignoring the features db path and train/val split ratio."
            f"Directly loading the features and erds instead."
        )
        validation_features, validation_erds = validate_features_erds
    else:
        assert (full_features_erds_db_path is not None) and (0 < training_split < 1.0)
        print(
            f"Loading the full database and splitting train/val data."
            f"Ensure that the same random seed is used for both this training function and the validation function."
        )

        features_all, erds_all = get_features_and_erds_from_db(Path(full_features_erds_db_path), save_type)
        [
            train_features,
            train_erds,
            validation_features,
            validation_erds,
        ] = training_validation_split(features_all, erds_all, training_split, random_seed=random_seed)

    if model_type != "slads-net":
        raise

    if training_split < 1:
        score = erd_model.model.score(validation_features, validation_erds)
        print("Validation score is", score)
    else:
        raise ValueError
    return score
