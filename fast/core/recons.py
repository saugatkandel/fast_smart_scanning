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
import numpy as np

from .data_structures import MeasurementInfo, NeighborsInfo


# Perform the reconstruction
def compute_recon(input_image, minfo: MeasurementInfo, neighbors: NeighborsInfo):
    # Create a blank image for the reconstruction
    recon_image = input_image.copy()  # np.zeros_like(input_image)

    # Compute reconstruction values using IDW (inverse distance weighting)
    if len(minfo.unmeasured_idxs) > 0:
        recon_image[
            minfo.unmeasured_idxs[:, 0], minfo.unmeasured_idxs[:, 1]
        ] = _compute_weighted_m_recons(neighbors)

    # Combine measured values back into the reconstruction image
    recon_image[
        minfo.measured_idxs[:, 0], minfo.measured_idxs[:, 1]
    ] = minfo.measured_values
    return recon_image


def _compute_weighted_m_recons(neighbors, recon_method="cwm"):
    # Weighted Mean Computation
    if recon_method == "cwm":
        recon_values = np.sum(neighbors.values * neighbors.weights, axis=1)
    elif recon_method == "dwm":
        recon_values = _compute_weighted_mode_recons(
            neighbors.values, neighbors.weights
        )
    else:
        raise ValueError
    return recon_values


def _compute_weighted_mode_recons(neighbors_values, neighbors_weights):
    # Weighted Mode Computation
    raise NotImplementedError
    class_labels = np.unique(neighbors_info.values)
    class_weight_sums = np.zeros(
        (np.shape(neighbors_info.weights)[0], np.shape(class_labels)[0])
    )
    for i in range(np.shape(class_labels)[0]):
        temp_feats = np.zeros(
            (np.shape(neighbors_info.weights)[0], np.shape(neighbors_info.weights)[1])
        )
        np.copyto(temp_feats, neighbors_info.weights)
        temp_feats[neighbors_info.values != class_labels[i]] = 0
        class_weight_sums[:, i] = np.sum(temp_feats, axis=1)
    idx_of_max_class = np.argmax(class_weight_sums, axis=1)
    recon_values = class_labels[idx_of_max_class]
    return recon_values
