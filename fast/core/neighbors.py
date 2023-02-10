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

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data_structures import MeasurementInfo, NeighborsInfo


def find_neighbors(minfo: MeasurementInfo, num_neighbors: int, resolution: int = 1):
    neigh = NearestNeighbors(n_neighbors=num_neighbors)
    neigh.fit(minfo.measured_idxs)
    neighbor_distances, neighbor_indices = neigh.kneighbors(minfo.unmeasured_idxs)
    neighbor_distances = neighbor_distances * resolution
    neighbor_values = minfo.measured_values[neighbor_indices]
    neighbor_weights = _compute_neighbor_weights(neighbor_distances)

    return NeighborsInfo(
        indices=neighbor_indices,
        distances=neighbor_distances,
        weights=neighbor_weights,
        values=neighbor_values,
    )


def _compute_neighbor_weights(neighbor_distances, power=2):
    """Calculating the weights for how each neighboring data point contributes
    to the reconstruction for the current location.

    First, the weights are calculated to be inversely proportional to the distance from teh current point.
    Next, the weights are normalized so that the total weight sums up to 1 for each reconstruction point."""

    unnormalized_weights = 1 / np.power(neighbor_distances, power)
    sum_over_row = np.sum(unnormalized_weights, axis=1, keepdims=True)
    weights = unnormalized_weights / sum_over_row
    return weights
