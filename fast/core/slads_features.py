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
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

from .data_structures import NeighborsInfo


# Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def compute_poly_features(
    sample_params,
    recon_image,
    update_idxs,
    neighbors: NeighborsInfo,
    feat_dist_cutoff: float,
    feat_type: str,
):
    # Retrieve recon values
    input_values = recon_image[update_idxs[:, 0], update_idxs[:, 1]]

    # Create array to hold features
    feature = np.zeros((np.shape(update_idxs)[0], 6))

    # Compute std div features
    diff_vec = compute_difference(neighbors.values, np.tile(input_values, (neighbors.values.shape[1], 1)).T)
    feature[:, 0] = np.sum(neighbors.weights * diff_vec, axis=1)
    feature[:, 1] = np.sqrt(np.sum(np.power(diff_vec, 2), axis=1))

    # Compute distance/density features
    cutoff_dist = np.ceil(np.sqrt((feat_dist_cutoff / 100) * (sample_params.image_size / np.pi)))
    feature[:, 2] = neighbors.distances[:, 0]
    neighbors_in_circle = np.sum(neighbors.distances <= cutoff_dist, axis=1)
    feature[:, 3] = (1 + (np.pi * (np.square(cutoff_dist)))) / (1 + neighbors_in_circle)

    # Compute gradient features; assume continuous features
    gradient_x, gradient_y = np.gradient(recon_image)
    feature[:, 4] = abs(gradient_y)[update_idxs[:, 0], update_idxs[:, 1]]
    feature[:, 5] = abs(gradient_x)[update_idxs[:, 0], update_idxs[:, 1]]

    # Fit polynomial features to the determined array
    if feat_type == "polynomial":
        return _compute_polynomial_features(feature)
    elif feat_type == "rbf":
        return _compute_rbf_features(feature)


# Determine absolute difference between two arrays
def compute_difference(array1, array2):
    return abs(array1 - array2)


def _compute_polynomial_features(feature):
    return PolynomialFeatures(degree=2).fit_transform(feature)


def _compute_rbf_features(feature):
    rbf_feature = RBFSampler(gamma=0.01, n_components=50, random_state=1)
    poly_features = rbf_feature.fit_transform(feature)
    return poly_features
