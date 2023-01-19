# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
from copy import deepcopy

import numpy as np
import numpy.typing as npt

from ..input_params import ERDInputParams, GeneralInputParams, SampleParams
from ..utils.renormalize import renormalize
from .data_structures import MeasurementInfo, NeighborsInfo
from .erd import ERDModel
from .measurement_interface import MeasurementInterface
from .neighbors import find_neighbors
from .recons import compute_recon
from .slads_features import compute_poly_features


class Sample:
    """Contains state information about the sample."""

    def __init__(
        self,
        sample_params: SampleParams,
        general_params: GeneralInputParams,
        erd_params: ERDInputParams,
        measurement_interface: MeasurementInterface,
        erd_model: ERDModel,
    ):

        self.params_sample = sample_params
        self.measurement_interface = measurement_interface
        # Initialize variables that are expected to exist

        self.recon_image = np.zeros(self.params_sample.image_shape)
        self.mask = np.zeros_like(self.recon_image)

        self.measurement_info = MeasurementInfo()
        self.neighbors = NeighborsInfo()

        self.RD = np.zeros_like(self.recon_image)
        self.ERD = np.zeros_like(self.RD)
        self.ratio_measured = 0

        self.params_gen = general_params
        self.params_erd = erd_params
        self.erd_model = erd_model

        self.iteration = 0

    def perform_measurements(self, new_idxs: npt.ArrayLike):

        new_idxs = np.atleast_2d(new_idxs)
        self.measurement_info.new_idxs = new_idxs
        self.mask[new_idxs[:, 0], new_idxs[:, 1]] = 1

        # Update which positions have not yet been measured
        self.measurement_info.measured_idxs = np.concatenate(
            (self.measurement_info.measured_idxs, new_idxs), axis=0
        )
        self.measurement_info.unmeasured_idxs = np.transpose(np.where(self.mask == 0))

        new_values = self.measurement_interface.perform_measurement(new_idxs)

        self.measurement_info.measured_values = np.concatenate(
            (self.measurement_info.measured_values, new_values)
        )

        # Update percentage pixels measured; only when not fromRecon
        self.ratio_measured = np.sum(self.mask) / self.params_sample.image_size

    @staticmethod
    def _perform_measurements_from_recon(
        mask: npt.NDArray,
        measurement_info: MeasurementInfo,
        recon_image: npt.NDArray,
        new_idxs: npt.ArrayLike,
    ) -> tuple[npt.NDArray, MeasurementInfo, npt.NDArray]:
        """
        Should only be called for the very specific situation that that we have a batch size > 1,
        so that we can update the ERDs using the reconstructed values. This should be a call from
        find_new_measurement_idxs
        """
        new_idxs = np.atleast_2d(new_idxs)
        new_values = recon_image[new_idxs[:, 0], new_idxs[:, 1]]
        mask[new_idxs[:, 0], new_idxs[:, 1]] = 1

        measurement_info.measured_idxs = np.concatenate(
            (measurement_info.measured_idxs, new_idxs), axis=0
        )
        measurement_info.unmeasured_idxs = np.transpose(np.where(mask == 0))
        measurement_info.measured_values = np.concatenate(
            (measurement_info.measured_values, new_values)
        )

        return mask, measurement_info, new_values

    def _get_limited_update_locations(
        self, mask: npt.NDArray, measurement_info: MeasurementInfo
    ) -> npt.NDArray[int]:

        if (
            self.params_erd.calculate_full_erd_per_step
            or self.iteration % self.params_erd.full_erd_recalculation_frequency == 0
        ):
            return measurement_info.unmeasured_idxs

        measured_idxs = measurement_info.measured_idxs
        new_idxs = measurement_info.new_idxs

        suggested_radius = np.sqrt(
            (1 / np.pi)
            * self.params_sample.image_size
            * self.params_gen.num_neighbors
            / measured_idxs.shape[0]
        )
        update_radius = np.max(
            [suggested_radius, self.params_erd.affected_neighbors_window_min]
        )
        update_radius = np.min(
            [update_radius, self.params_erd.affected_neighbors_window_max]
        ).astype("int")

        update_radius_mat = np.zeros(self.params_sample.image_shape)

        done = False
        unmeasured_idxs = []
        while not done:
            for ix, iy in new_idxs:
                uix1 = max(ix - update_radius, 0)
                uix2 = int(min(ix + update_radius, self.params_sample.image_shape[0]))
                uiy1 = max(iy - update_radius, 0)
                uiy2 = min(iy + update_radius, self.params_sample.image_shape[1])

                update_radius_mat[uix1:uix2, uiy1:uiy2] = 1

            # update_idxs = np.where(update_radius_mat[~self.mask] == 1)
            unmeasured_idxs = np.transpose(
                np.where(np.logical_and(mask == 0, update_radius_mat == 1))
            )

            if unmeasured_idxs.size == 0:
                update_radius = (
                    update_radius * self.params_erd.affected_window_increase_factor
                )
            else:
                done = True
        return unmeasured_idxs

    def reconstruct_and_compute_erd(self):
        # Compute feature information for SLADS models; not needed for DLADS
        unmeasured_idxs = self._get_limited_update_locations(
            self.mask.copy(), deepcopy(self.measurement_info)
        )

        measurement_info = deepcopy(self.measurement_info)
        measurement_info.unmeasured_idxs = unmeasured_idxs

        # Determine neighbor information for all unmeasured locations
        if len(self.measurement_info.unmeasured_idxs) > 0:
            self.neighbors = find_neighbors(
                measurement_info, self.params_gen.num_neighbors
            )
        else:
            self.neighbors = NeighborsInfo([], [], [], [])

        # Compute reconstructions, resize to physical dimensions
        self.recon_image = compute_recon(
            self.recon_image, measurement_info, self.neighbors
        )

        if self.params_erd.model_type == "slads-net":
            self.poly_features = compute_poly_features(
                self.params_sample,
                self.recon_image,
                unmeasured_idxs,
                self.neighbors,
                self.params_erd.feat_distance_cutoff,
                self.params_erd.feature_type,
            )

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion

        if len(measurement_info.unmeasured_idxs) == 0:
            self.RD = np.zeros(self.params_sample.image_shape)
            self.ERD = np.zeros(self.params_sample.image_shape)
            return

        # Compute the ERD with the prescribed model
        if self.params_erd.model_type == "slads-net":
            erds = self.erd_model.predict(self.poly_features)

        self.ERD[unmeasured_idxs[:, 0], unmeasured_idxs[:, 1]] = erds
        self.ERD = self._rescale_and_fix_erd(self.ERD, self.mask)
        self.iteration += 1

    def _reconstruct_and_compute_erd_from_recon(
        self, mask, measurement_info, recon_image, erd
    ):
        # Compute feature information for SLADS models; not needed for DLADS

        unmeasured_idxs = self._get_limited_update_locations(
            mask.copy(), deepcopy(measurement_info)
        )

        measurement_info = deepcopy(measurement_info)
        measurement_info.unmeasured_idxs = unmeasured_idxs

        # Determine neighbor information for unmeasured locations
        neighbors = find_neighbors(measurement_info, self.params_gen.num_neighbors)

        # Compute reconstructions, resize to physical dimensions
        recon_image = compute_recon(recon_image, measurement_info, neighbors)

        if self.params_erd.model_type == "slads-net":
            poly_features = compute_poly_features(
                self.params_sample,
                recon_image,
                unmeasured_idxs,
                neighbors,
                self.params_erd.feat_distance_cutoff,
                self.params_erd.feature_type,
            )

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion

        if len(measurement_info.unmeasured_idxs) == 0:
            erd = np.zeros(self.params_sample.image_shape)
            return erd

        # Compute the ERD with the prescribed model
        if self.params_erd.model_type == "slads-net":
            erd_this = self.erd_model.predict(poly_features)

        erd[unmeasured_idxs[:, 0], unmeasured_idxs[:, 1]] = erd_this
        erd = self._rescale_and_fix_erd(erd, mask)
        return recon_image, erd

    @staticmethod
    def _rescale_and_fix_erd(erd, mask):
        erd = erd.copy()
        erd[erd < 0] = 0
        erd = erd * (1 - mask)
        erd = np.nan_to_num(erd, nan=0, posinf=0, neginf=0)
        # if erd.max() != 0:
        #    erd = (erd - erd.min()) / (erd.max() - erd.min())
        return erd

    def _iterative_find_new_idxs_from_recon(self):
        # erd_values = self.ERD[self.measurement_info.unmeasured_idxs[:, 0],
        #                      self.measurement_info.unmeasured_idxs[:, 1]]
        # max_k_indices = np.argpartition(erd_values, -outer_batch_size)[-outer_batch_size:]
        # return self.measurement_info.unmeasured_idxs[max_k_indices]

        # Create a list to hold the chosen scanning locations
        measurement_info = deepcopy(self.measurement_info)
        mask = self.mask.copy()
        erd = self.ERD.copy()
        recon_image = self.recon_image.copy()
        new_idxs = []

        # Until the percToScan has been reached, substitute reconstruction values for actual measurements
        while True:
            unmeasured_idxs = measurement_info.unmeasured_idxs
            # If there are no more points with physical ERD > 0, break from loop
            if np.sum(erd) <= 0:
                break

            # Find next measurement location and store the chosen location for later, actual measurement
            new_idx = unmeasured_idxs[
                np.argmax(erd[unmeasured_idxs[:, 0], unmeasured_idxs[:, 1]])
            ]

            # Perform the measurement, using values from reconstruction
            mask, measurement_info, new_values = self._perform_measurements_from_recon(
                mask, measurement_info, recon_image, new_idx
            )
            recon_image, erd = self._reconstruct_and_compute_erd_from_recon(
                mask, measurement_info, recon_image, erd
            )

            # When enough new locations have been determined, break from loop
            new_idxs.append(new_idx.tolist())
            if len(new_idxs) >= self.params_sample.outer_batch_size:
                break
        return new_idxs

    # Determine which unmeasured points of a sample should be scanned given the current E/RD
    def find_new_measurement_idxs(self):
        if self.params_sample.scan_method == "random":
            np.random.shuffle(self.measurement_info.unmeasured_idxs)
            new_idxs = self.measurement_info.unmeasured_idxs[
                : self.params_sample.outer_batch_size
            ].astype(int)

        elif self.params_sample.scan_method == "pointwise":
            batch_size = self.params_sample.outer_batch_size
            # If performing a groupwise scan, use reconstruction as the measurement value,
            # until reaching target number of points to scan
            if batch_size > 1:
                new_idxs = self._iterative_find_new_idxs_from_recon()
            else:
                # Identify the unmeasured location with the highest physicalERD value;
                # return in a list to ensure it is iterable
                inb = self.params_sample.inner_batch_size
                erd_values = self.ERD[
                    self.measurement_info.unmeasured_idxs[:, 0],
                    self.measurement_info.unmeasured_idxs[:, 1],
                ]
                max_k_indices = np.argpartition(erd_values, -inb)[-inb:]

                new_idxs = self.measurement_info.unmeasured_idxs[max_k_indices]
                new_values = self.ERD[new_idxs[:, 0], new_idxs[:, 1]]
                sorted_idxs = np.argsort(-new_values)
                new_idxs = new_idxs[sorted_idxs]

        return np.asarray(new_idxs)

    def perform_initial_scan(self):
        self.perform_measurements(self.params_sample.initial_idxs)
        self.reconstruct_and_compute_erd()


class ExperimentalSample(Sample):
    """Contains state information about the sample."""

    def perform_measurements(self, new_idxs):

        new_idxs = np.atleast_2d(new_idxs)
        self.measurement_info.new_idxs = new_idxs
        self.mask[new_idxs[:, 0], new_idxs[:, 1]] = 1

        # Update which positions have not yet been measured
        self.measurement_info.measured_idxs = np.concatenate(
            (self.measurement_info.measured_idxs, new_idxs), axis=0
        )
        self.measurement_info.unmeasured_idxs = np.transpose(np.where(self.mask == 0))

        new_values_before_norm = self.measurement_interface.perform_measurement(
            new_idxs
        )
        self.measurement_info.unnormalized_values = np.concatenate(
            (self.measurement_info.unnormalized_values, new_values_before_norm)
        )

        # self.measurement_info.measured_values = np.concatenate((self.measurement_info.measured_values, new_values))
        self.measurement_info.measured_values = renormalize(
            self.measurement_info.unnormalized_values
        )
        # self.measurement_info.measured_values = self.measurement_info.unnormalized_values

        # Update percentage pixels measured; only when not fromRecon
        self.ratio_measured = np.sum(self.mask) / self.params_sample.image_size
