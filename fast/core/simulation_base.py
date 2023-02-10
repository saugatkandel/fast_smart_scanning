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

from copy import deepcopy

import numpy as np
import scipy.signal

from ..input_params import SimulatedSampleParams
from .base import Sample
from .data_structures import NeighborsInfo
from .neighbors import find_neighbors
from .recons import compute_recon
from .slads_features import compute_poly_features


class SimulatedSample(Sample):
    def __init__(self, sample_params: SimulatedSampleParams, *args: int, **kwargs: int):
        super().__init__(sample_params, *args, **kwargs)

        self.RDPP = np.zeros_like(self.recon_image)

    def reconstruct_and_compute_erd(self):
        # calculate_full_erd recalculates the RD for the full image and not just around the
        # new update location.

        if self.params_sample.simulation_type == "emulation":
            super().reconstruct_and_compute_erd()
            return

        if len(self.measurement_info.unmeasured_idxs) > 0:
            self.neighbors = find_neighbors(self.measurement_info, self.params_gen.num_neighbors)
        else:
            self.neighbors = NeighborsInfo([], [], [], [])

        # Compute reconstructions, resize to physical dimensions and average for visualization
        self.recon_image = compute_recon(self.recon_image, self.measurement_info, self.neighbors)

        if self.params_erd.model_type == "slads-net":
            self.poly_features = compute_poly_features(
                self.params_sample,
                self.recon_image,
                self.measurement_info.unmeasured_idxs,
                self.neighbors,
                self.params_erd.feat_distance_cutoff,
                self.params_erd.feature_type,
            )

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion

        if len(self.measurement_info.unmeasured_idxs) == 0:
            self.RD = np.zeros(self.params_sample.image_size)
            self.ERD = np.zeros(self.params_sample.image_size)

        # If this is a full measurement step, compute the RDPP
        self.RDPP = np.abs(self.params_sample.image - self.recon_image)

        # Compute the RD and use it in place of an ERD;
        # only save times if they are fully computed, not just being updated
        unmeasured_idxs = self._get_limited_update_locations(self.mask.copy(), deepcopy(self.measurement_info))
        self.RD = self._compute_rd(
            self.mask,
            self.measurement_info,
            self.neighbors,
            self.RDPP,
            self.ERD,
            unmeasured_idxs,
        )
        self.ERD = self.RD.copy()
        self.ERD = self._rescale_and_fix_erd(self.ERD, self.mask)
        self.iteration += 1

    def _reconstruct_and_compute_erd_from_recon(self, mask, measurement_info, recon_image, erd):
        # Compute feature information for SLADS models; not needed for DLADS
        if self.params_sample.simulation_type == "emulation":
            return super()._reconstruct_and_compute_erd_from_recon(mask, measurement_info, recon_image, erd)
        if self.params_sample.simulation_type != "visualize_erd":
            raise NotImplementedError("Batch size > 1 not supported for training run")

        unmeasured_idxs = self._get_limited_update_locations(mask.copy(), deepcopy(measurement_info))

        measurement_info = deepcopy(measurement_info)
        measurement_info.unmeasured_idxs = unmeasured_idxs

        # Determine neighbor information for unmeasured locations
        neighbors = find_neighbors(measurement_info, self.params_gen.num_neighbors)

        # Compute reconstructions, resize to physical dimensions
        recon_image = compute_recon(recon_image, measurement_info, self.neighbors)

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion
        if len(measurement_info.unmeasured_idxs) == 0:
            erd = np.zeros(self.params_sample.image_shape)
            return erd
        # If this is a full measurement step, compute the RDPP
        rdpp = np.abs(self.params_sample.image - recon_image)

        # update_idxs = self._get_limited_update_locations(mask, measurement_info)
        rd = self._compute_rd(mask, measurement_info, neighbors, rdpp, erd, unmeasured_idxs)
        erd = self._rescale_and_fix_erd(rd, mask)
        return recon_image, erd

    # Compute approximated Reduction in Distortion (RD) values
    def _compute_rd(self, mask, measurement_info, neighbors, rdpp, rd, update_idxs):
        # If a full calculation of RD then use the unmeasured locations,
        # otherwise find those that should be updated based on the latest measurement idxs

        indices = [np.where(np.all(measurement_info.unmeasured_idxs == idx, axis=1))[0][0] for idx in update_idxs]

        closest_neighbor_distances = neighbors.distances[:, 0][indices]

        # Calculate the sigma values for chosen c value
        sigma_values = closest_neighbor_distances / self.params_erd.c_value

        # Determine window sizes, ensuring odd values, and radii for Gaussian generation
        if not self.params_erd.static_window:
            window_sizes = np.ceil(2 * self.params_erd.dynamic_window_sigma_mult * sigma_values).astype(int) + 1
        else:
            window_sizes = (np.ones((len(sigma_values))) * self.params_erd.static_window_size).astype(int)

        # window_sizes[window_sizes < self.params_l1.min_window_size] = self.params_l1.min_window_size
        # window_sizes[window_sizes > self.params_l1.max_window_size] = self.params_l1.max_window_size
        window_sizes[window_sizes % 2 == 0] += 1

        radii = (window_sizes // 2).reshape(-1, 1).astype(int)

        # Zero-pad the RDPP and get_by_index offset positions according to maxRadius for window extraction
        max_radius = np.max(radii)
        padded_rdpps = np.pad(rdpp, [(max_radius, max_radius), (max_radius, max_radius)], mode="constant")
        offset_locations = update_idxs + max_radius
        start_offsets, stop_offsets = offset_locations - radii, offset_locations + radii

        gaussian_windows = {}
        rds_temp = rd.copy()
        for ix in range(update_idxs.shape[0]):
            sig_this = sigma_values[ix]
            win_this = window_sizes[ix]
            s1 = start_offsets[ix]
            s2 = stop_offsets[ix]
            if (sig_this, win_this) not in gaussian_windows:
                gaussian_signal = scipy.signal.gaussian(window_sizes[ix], sig_this)
                gaussian_window = np.outer(gaussian_signal, gaussian_signal)
                gaussian_windows[(sig_this, win_this)] = gaussian_window

            update_mat = padded_rdpps[s1[0] : s2[0] + 1, s1[1] : s2[1] + 1] * gaussian_windows[(sig_this, win_this)]
            rds_temp[update_idxs[ix, 0], update_idxs[ix, 1]] = np.sum(update_mat)

        # Make sure measured locations have 0 RD values
        rd = rds_temp * (1 - mask)
        return rd
