import numpy as np

from .data_structures import NeighborsInfo, MeasurementInfo


# Perform the reconstruction
def compute_recon(input_image, minfo: MeasurementInfo, neighbors: NeighborsInfo):
    # Create a blank image for the reconstruction
    recon_image = input_image.copy()  # np.zeros_like(input_image)

    # Compute reconstruction values using IDW (inverse distance weighting)
    if len(minfo.unmeasured_idxs) > 0:
        recon_image[minfo.unmeasured_idxs[:, 0], minfo.unmeasured_idxs[:, 1]] = _compute_weighted_m_recons(neighbors)

    # Combine measured values back into the reconstruction image
    recon_image[minfo.measured_idxs[:, 0], minfo.measured_idxs[:, 1]] = minfo.measured_values
    return recon_image


def _compute_weighted_m_recons(neighbors, recon_method='cwm'):
    # Weighted Mean Computation
    if recon_method == 'cwm':
        recon_values = np.sum(neighbors.values * neighbors.weights, axis=1)
    elif recon_method == 'dwm':
        recon_values = _compute_weighted_mode_recons(neighbors.values, neighbors.weights)
    else:
        raise ValueError
    return recon_values


def _compute_weighted_mode_recons(neighbors_values, neighbors_weights):
    # Weighted Mode Computation
    raise NotImplementedError
    class_labels = np.unique(neighbors_info.values)
    class_weight_sums = np.zeros((np.shape(neighbors_info.weights)[0], np.shape(class_labels)[0]))
    for i in range(np.shape(class_labels)[0]):
        temp_feats = np.zeros((np.shape(neighbors_info.weights)[0], np.shape(neighbors_info.weights)[1]))
        np.copyto(temp_feats, neighbors_info.weights)
        temp_feats[neighbors_info.values != class_labels[i]] = 0
        class_weight_sums[:, i] = np.sum(temp_feats, axis=1)
    idx_of_max_class = np.argmax(class_weight_sums, axis=1)
    recon_values = class_labels[idx_of_max_class]
    return recon_values
