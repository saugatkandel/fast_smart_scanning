import abc
import dataclasses as dt
import time

import numpy as np
import scipy.signal

from .data_structures import NeighborsInfo, MeasurementInfo
from .erd import ERDModel
from .measurement_interface import MeasurementInterface
from .neighbors import find_neighbors
from .recons import compute_recon
from .slads_features import compute_poly_features
from ..input_params import GeneralInputParams, ERDInputParams, SampleParams

SIMULATION_OPTIONS = ['training', 'c_opt']


class Sample:
    """Contains state information about the sample."""

    def __init__(self, sample_params: SampleParams,
                 general_params: GeneralInputParams,
                 erd_params: ERDInputParams,
                 measurement_interface: MeasurementInterface,
                 erd_model: ERDModel):

        self.params_sample = sample_params
        self.measurement_interface = measurement_interface
        # Initialize variables that are expected to exist

        self.recon_image = np.zeros(self.params_sample.image_shape)
        self.mask = np.zeros_like(self.recon_image)

        self.measurement_info = MeasurementInfo(measured_idxs=np.empty((0, 2), dtype='int'),
                                                unmeasured_idxs=np.empty((0, 2), dtype='int'),
                                                measured_values=np.empty(0, dtype='float32'),
                                                new_idxs=[])
        self.neighbors = NeighborsInfo([], [], [], [])

        self.RD = np.zeros_like(self.recon_image)
        self.ERD = np.zeros_like(self.RD)
        self.ratio_measured = 0

        self.params_gen = general_params
        self.params_erd = erd_params
        self.erd_model = erd_model

        self.iteration = 0

    def perform_measurements(self, new_idxs):

        new_idxs = np.atleast_2d(new_idxs)
        self.measurement_info.new_idxs = new_idxs
        self.mask[new_idxs[:, 0], new_idxs[:, 1]] = 1

        # Update which positions have not yet been measured
        self.measurement_info.measured_idxs = np.concatenate((self.measurement_info.measured_idxs,
                                                              new_idxs), axis=0)
        self.measurement_info.unmeasured_idxs = np.transpose(np.where(self.mask == 0))

        new_values = self.measurement_interface.perform_measurement(new_idxs)

        self.measurement_info.measured_values = np.concatenate((self.measurement_info.measured_values, new_values))

        # Update percentage pixels measured; only when not fromRecon
        self.ratio_measured = (np.sum(self.mask) / self.params_sample.image_size)

    def _get_limited_update_locations(self):
        measured_idxs = self.measurement_info.measured_idxs
        new_idxs = self.measurement_info.new_idxs

        suggested_radius = np.sqrt((1 / np.pi) * self.params_sample.image_size *
                                   self.params_gen.num_neighbors / measured_idxs.shape[0])
        update_radius = np.max([suggested_radius, self.params_erd.affected_neighbors_window_min])
        update_radius = np.min([update_radius, self.params_erd.affected_neighbors_window_max]).astype('int')

        update_radius_mat = np.zeros(self.params_sample.image_shape)

        done = False
        while not done:
            for ix, iy in new_idxs:
                uix1 = max(ix - update_radius, 0)
                uix2 = int(min(ix + update_radius, self.params_sample.image_shape[0]))
                uiy1 = max(iy - update_radius, 0)
                uiy2 = min(iy + update_radius, self.params_sample.image_shape[1])

                update_radius_mat[uix1: uix2, uiy1: uiy2] = 1

            # update_idxs = np.where(update_radius_mat[~self.mask] == 1)
            unmeasured_idxs = np.transpose(np.where(np.logical_and(self.mask == 0, update_radius_mat == 1)))

            if unmeasured_idxs.size == 0:
                update_radius = update_radius * self.params_erd.affected_window_increase_factor
            else:
                done = True
        return unmeasured_idxs

    def reconstruct_and_compute_erd(self):
        # Compute feature information for SLADS models; not needed for DLADS

        if (not self.params_erd.calculate_full_erd_per_step) and (self.iteration > 0):
            unmeasured_idxs = self._get_limited_update_locations()
        else:
            unmeasured_idxs = self.measurement_info.unmeasured_idxs

        measurement_info = dt.replace(self.measurement_info, unmeasured_idxs=unmeasured_idxs)
        # Determine neighbor information for unmeasured locations
        if len(self.measurement_info.unmeasured_idxs) > 0:
            self.neighbors = find_neighbors(measurement_info, self.params_gen.num_neighbors)
        else:
            self.neighbors = NeighborsInfo([], [], [], [])
        # print('Neighbors_time:', time.time() - t1)

        # Compute reconstructions, resize to physical dimensions and average for visualization
        self.recon_image = compute_recon(self.recon_image, measurement_info, self.neighbors)
        # print('Recon time', time.time() - t1)

        if self.params_erd.model_type == 'slads-net':
            self.poly_features = compute_poly_features(self.params_sample, self.recon_image,
                                                       unmeasured_idxs, self.neighbors,
                                                       self.params_erd.feat_distance_cutoff,
                                                       self.params_erd.feature_type)

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion

        if len(self.measurement_info.unmeasured_idxs) == 0:
            self.RD = np.zeros(self.params_sample.image_shape)
            self.ERD = np.zeros(self.params_sample.image_shape)
            return

        # Compute the ERD with the prescribed model
        if self.params_erd.model_type == 'slads-net':
            erds = self.erd_model.predict(self.poly_features)

        self.ERD[unmeasured_idxs[:, 0], unmeasured_idxs[:, 1]] = erds
        self._rescale_and_fix_erd()
        self.iteration += 1

    def _rescale_and_fix_erd(self):
        self.ERD[self.ERD < 0] = 0
        self.ERD = self.ERD * (1 - self.mask)
        self.ERD = np.nan_to_num(self.ERD, nan=0, posinf=0, neginf=0)
        if self.ERD.max() != 0:
            self.ERD = (self.ERD - self.ERD.min()) / (self.ERD.max() - self.ERD.min())

    # Determine which unmeasured points of a sample should be scanned given the current E/RD
    def find_new_measurement_idxs(self):
        if self.params_sample.scan_method == 'random':
            np.random.shuffle(self.measurement_info.unmeasured_idxs)
            new_idxs = self.measurement_info.unmeasured_idxs[:self.params_sample.points_to_scan].astype(int)
            return new_idxs
        elif self.params_sample.scan_method == 'pointwise':
            bsize = self.params_sample.points_to_scan
            # If performing a groupwise scan, use reconstruction as the measurement value,
            # until reaching target number of points to scan
            if bsize > 1:
                erd_values = self.ERD[self.measurement_info.unmeasured_idxs[:, 0],
                                      self.measurement_info.unmeasured_idxs[:, 1]]
                max_k_indices = np.argpartition(erd_values, -bsize)[-bsize:]
                return self.measurement_info.unmeasured_idxs[max_k_indices]

                # Create a list to hold the chosen scanning locations
                newIdxs = []

                # Until the percToScan has been reached, substitute reconstruction values for actual measurements
                while True:

                    # If there are no more points with physical ERD > 0, break from loop
                    if np.sum(self.physicalERD) <= 0: break

                    # Find next measurement location and store the chosen location for later, actual measurement
                    newIdx = self.unmeasured_idxs[
                        np.argmax(self.physicalERD[self.unmeasured_idxs[:, 0], self.unmeasured_idxs[:, 1]])]
                    newIdxs.append(newIdx.tolist())

                    # Perform the measurement, using values from reconstruction
                    self.performMeasurements(self.params_sample, result, newIdx, model, cValue, bestCFlag, oracleFlag,
                                             datagenFlag, True)

                    # When enough new locations have been determined, break from loop
                    if (np.sum(self.mask) - np.sum(result.lastMask)) >= self.params_sample.pointsToScan: break

                # Convert to array for indexing
                newIdxs = np.asarray(newIdxs)
            else:
                # Identify the unmeasured location with the highest physicalERD value;
                # return in a list to ensure it is iterable
                new_idxs = np.asarray([self.measurement_info.unmeasured_idxs[
                                           np.argmax(self.ERD[
                                                         self.measurement_info.unmeasured_idxs[:, 0],
                                                         self.measurement_info.unmeasured_idxs[:, 1]])].tolist()])

        return new_idxs


class SimulatedSample(Sample):
    def __init__(self, simulation_type: str, *args: int, **kwargs: int):
        assert simulation_type in SIMULATION_OPTIONS
        self.simulation_type = simulation_type
        super().__init__(*args, **kwargs)

        self.RDPP = np.zeros_like(self.recon_image)

    def reconstruct_and_compute_erd(self):
        # calculate_full_erd recalculates the RD for the full image and not just around the
        # new update location.

        if self.simulation_type != 'training':
            super().reconstruct_and_compute_erd()
            return

        if len(self.measurement_info.unmeasured_idxs) > 0:
            self.neighbors = find_neighbors(self.measurement_info, self.params_gen.num_neighbors)
        else:
            self.neighbors = NeighborsInfo([], [], [], [])

        # Compute reconstructions, resize to physical dimensions and average for visualization
        self.recon_image = compute_recon(self.recon_image, self.measurement_info, self.neighbors)

        if self.params_erd.model_type == 'slads-net':
            self.poly_features = compute_poly_features(self.params_sample, self.recon_image,
                                                       self.measurement_info.unmeasured_idxs, self.neighbors,
                                                       self.params_erd.feat_distance_cutoff,
                                                       self.params_erd.feature_type)

        # Compute RD/ERD; if every location has been scanned all positions are zero
        # Determine the Estimated Reduction in Distortion

        if len(self.measurement_info.unmeasured_idxs) == 0:
            self.RD = np.zeros(self.params_sample.image_size)
            self.ERD = np.zeros(self.params_sample.image_size)

        # If this is a full measurement step, compute the RDPP
        self.RDPP = np.abs(self.params_sample.image - self.recon_image)

        # Compute the RD and use it in place of an ERD;
        # only save times if they are fully computed, not just being updated
        t1 = time.time()
        self._compute_rd()
        # print('RD time', time.time()- t1)
        self.ERD = self.RD
        self._rescale_and_fix_erd()
        self.iteration += 1

    # Compute approximated Reduction in Distortion (RD) values
    def _compute_rd(self):
        # If a full calculation of RD then use the unmeasured locations,
        # otherwise find those that should be updated based on the latest measurement idxs

        if self.params_erd.calculate_full_erd_per_step:
            update_idxs = self.measurement_info.unmeasured_idxs
            closest_neighbor_distances = self.neighbors.distances[:, 0]
        else:
            raise NotImplementedError("Currently not supported")
            measured_idxs = self.measurement_info.measured_idxs
            closest_neighbor_indices = self.neighbors.indices[:, 0]

            # Find indices of new_idxs and then the indices of closest neighboring unmeasured location
            indices1 = [np.where(np.all(measured_idxs == idx, axis=1))[0]
                        for idx in self.measurement_info.new_idxs]

            # Update only if an unmeasured location has a freshly measured location as its closest measured neighbor
            indices2 = np.concatenate([np.argwhere(closest_neighbor_indices == idx) for idx in indices1]).flatten()

            # If there are no locations that need updating, then just return
            if len(indices2) == 0:
                return

            # Extract unmeasured idxs to be updated and their relevant neighbor information (to avoid recalculation)
            update_idxs = self.measurement_info.unmeasured_idxs[indices2]
            closest_neighbor_distances = self.neighbors.distances[:, 0][indices2]

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
        padded_rdpps = np.pad(self.RDPP, [(max_radius, max_radius), (max_radius, max_radius)], mode='constant')
        offset_locations = update_idxs + max_radius
        start_offsets, stop_offsets = offset_locations - radii, offset_locations + radii

        gaussian_windows = {}
        rds_temp = self.RD.copy()
        for ix in range(update_idxs.shape[0]):
            sig_this = sigma_values[ix]
            win_this = window_sizes[ix]
            s1 = start_offsets[ix]
            s2 = stop_offsets[ix]
            if (sig_this, win_this) not in gaussian_windows:
                gaussian_signal = scipy.signal.gaussian(window_sizes[ix], sig_this)
                gaussian_window = np.outer(gaussian_signal, gaussian_signal)
                gaussian_windows[(sig_this, win_this)] = gaussian_window

            update_mat = padded_rdpps[s1[0]: s2[0] + 1, s1[1]: s2[1] + 1] * gaussian_windows[(sig_this, win_this)]
            rds_temp[update_idxs[ix, 0], update_idxs[ix, 1]] = np.sum(update_mat)

        self.RD = rds_temp

        # Make sure measured locations have 0 RD values
        self.RD *= (1 - self.mask)
