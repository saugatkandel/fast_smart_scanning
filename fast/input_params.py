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
import dataclasses as dt

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

# from skopt.sampler import Hammersly
# from skopt.space import Space

SUPPORTED_FEATURE_TYPES = ["polynomial", "rbf"]
SUPPORTED_MODEL_TYPES = ["slads-net"]
SUPPORTED_SAMPLING_TYPES = ["slow_detailed", "fast_limited"]
SCAN_TYPES = ["transmission", "diffraction"]
GRID_TO_PIXEL_METHODS = ["pixel", "subpixel"]
SIMULATION_TYPES = ["training", "emulation", "visualize_erd"]
SUPPORTED_INITIAL_MASK_TYPES = ["random", "hammersly", "sobol", "halton", "latin-hypercube"]
SUPPORTED_SAVE_DATABASE_TYPES = ["slads-net-reduced", "generic"]


@dt.dataclass(frozen=True)
class TrainingInputParams:
    initial_scan_ratio: float = 0.01  # initial scan ratio
    initial_mask_type: str = "random"
    stop_ratio: float = 0.8  # stop ratio
    scan_method: str = "random"  # pointwise or line scan
    scan_type: str = "transmission"
    sampling_type: str = "fast_limited"
    num_repeats_per_mask: int = 1
    measurements_per_initial_mask: int = 10  # Only used if sampling type is fast_limited
    random_seed: int = 111
    training_split: float = 0.9
    test_c_values: list = dt.field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    calculate_full_erd_per_step: bool = True
    save_type: str = "slads-net-reduced"
    verbose_training: bool = True

    def __post_init__(self):
        assert self.sampling_type in SUPPORTED_SAMPLING_TYPES
        assert self.save_type in SUPPORTED_SAVE_DATABASE_TYPES


@dt.dataclass(frozen=True)
class SladsModelParams:
    random_state: int = 1
    activation: str = "relu"
    hidden_layer_sizes: tuple = (50, 5)
    solver: str = "adam"
    max_iter: int = 500
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3


@dt.dataclass(frozen=True)
class ERDInputParams:
    c_value: float
    model_type: str = "slads-net"
    static_window: bool = False
    static_window_size: int = 15
    dynamic_window_sigma_mult: float = 3
    feat_distance_cutoff: float = 0.25
    feature_type: str = "rbf"
    calculate_full_erd_per_step: bool = False
    full_erd_recalculation_frequency: int = 20
    affected_neighbors_window_min: float = 10
    affected_neighbors_window_max: float = 20
    affected_window_increase_factor: float = 1.5

    def __post_init__(self):
        assert self.model_type in SUPPORTED_MODEL_TYPES
        assert self.feature_type in SUPPORTED_FEATURE_TYPES


@dt.dataclass(frozen=True)
class GeneralInputParams:
    desired_td: float = 0
    num_neighbors: int = 10
    neighbor_weight_distance_norm: float = 2.0


@dt.dataclass
class SampleParams:
    """Contains information that is held constant throughout the procedure."""

    image_shape: tuple[int, int]  # tuple containing image shape
    initial_scan_ratio: float = 0.01  # initial scan ratio
    initial_scan_points_num: int = None
    stop_ratio: float = 0.4  # stop ratio
    scan_method: str = "pointwise"  # pointwise or random scan
    scan_type: str = "transmission"
    outer_batch_size: int = 1
    inner_batch_size: int = 1
    random_seed: int = None
    initial_idxs: list = None
    initial_mask_type: str = "halton"
    initial_mask: np.ndarray = dt.field(init=False)
    line_revisit: bool = False  # not in use right now
    rng: np.random.Generator = None
    image_size: int = dt.field(init=False)

    def __post_init__(self):
        assert self.initial_mask_type in SUPPORTED_INITIAL_MASK_TYPES
        if self.rng is not None:
            if self.random_seed is not None:
                raise ValueError("Cannot supply both rng and random seed.")
        else:
            self.rng = np.random.default_rng(self.random_seed)

        self.image_size = self.image_shape[0] * self.image_shape[1]
        if self.initial_idxs is None:
            if self.initial_scan_ratio is not None and self.initial_scan_points_num is not None:
                raise ValueError("Can only supply one of these.")
            if self.initial_scan_ratio is not None:
                self.initial_scan_points_num = int(self.initial_scan_ratio * self.image_size)
            else:
                self.initial_scan_ratio = self.initial_scan_points_num / self.image_size

            self.initial_idxs, self.initial_mask = self.generate_initial_mask(self.scan_method)
        else:
            self.initial_mask = np.zeros(self.image_shape)
            self.initial_mask[self.initial_idxs[:, 0], self.initial_idxs[:, 1]] = 1

        if self.scan_type != "transmission":
            raise NotImplementedError

    def generate_initial_mask(self, scan_method):
        # Update the scan method
        self.scan_method = scan_method

        # List of what points/lines should be initially measured

        # If scanning with line-bounded constraint
        if self.scan_method not in ["linewise", "pointwise", "random"]:
            raise NotImplementedError

        if self.scan_method == "linewise":
            new_idxs, new_mask = self._gen_linewise_scan()
        print(self.scan_method, self.initial_mask_type)
        # scan_method is eithe rpointwise or random
        match self.initial_mask_type:
            case "random":
                new_idxs, new_mask = self._gen_pointwise_scan()
            case "hammersly":
                new_idxs, new_mask = self._gen_hammersly_scan()
            case "sobol" | "halton" | "latin-hypercube":
                new_idxs, new_mask = self._gen_scipy_quasi_monte_carlo_scan()
            case _:
                raise NotImplementedError

        return new_idxs, new_mask

    def _gen_pointwise_scan(self):
        # Randomly select points for the initial scan
        mask = self.rng.random(self.image_shape)
        mask = mask <= self.initial_scan_ratio
        new_idxs = np.array(np.where(mask)).T
        return new_idxs, mask

    def _gen_hammersly_scan(self):
        # Use the Hammersly sequence to generate the low discrepancy random scan points

        raise NotImplementedError(
            "Scikit optimize has not been updated to keep up with new >1.24 typing conventions, \
            and using it with newer numpy versions gives errors. Disabling the Hammersely \
            sequence until this issue is resolved."
        )

        seed = self.rng.integers(1000)

        space = Space([[0, self.image_shape[0] - 1], [0, self.image_shape[1] - 1]])
        sampler = Hammersly()
        new_idxs = np.array(sampler.generate(space, self.initial_scan_points_num, random_state=seed))
        mask = np.zeros(self.image_shape, dtype="bool")
        mask[new_idxs[:, 0], new_idxs[:, 1]] = 1
        return new_idxs, mask

    def _gen_scipy_quasi_monte_carlo_scan(self):
        seed = self.rng.integers(1000)
        match self.initial_mask_type:
            case "sobol":
                sampler = qmc.Sobol(d=2, seed=seed)
            case "halton":
                sampler = qmc.Halton(d=2, seed=seed)
            case "latin-hypercube":
                sampler = qmc.LatinHypercube(d=2, seed=seed)

        samples = sampler.integers(
            n=self.initial_scan_points_num,
            l_bounds=[0, 0],
            u_bounds=[self.image_shape[0], self.image_shape[1]],
        )
        mask = np.zeros(self.image_shape, dtype="bool")
        mask[samples[:, 0], samples[:, 1]] = 1
        return samples, mask

    def _gen_linewise_scan(self):
        raise NotImplementedError


class SimulatedSampleParams(SampleParams):
    def __init__(self, image: npt.NDArray, simulation_type: str, *args, **kwargs):
        assert simulation_type in SIMULATION_TYPES
        self.image = image
        self.simulation_type = simulation_type
        super().__init__(image.shape, *args, **kwargs)


class ExperimentSampleParams(SampleParams):
    def __init__(self, image: npt.NDArray, simulation_type: str, *args, **kwargs):
        assert simulation_type in SIMULATION_TYPES
        self.image = image
        self.simulation_type = simulation_type
        super().__init__(image.shape, *args, **kwargs)
