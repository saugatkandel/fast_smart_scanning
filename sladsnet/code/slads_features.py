import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

from .data_structures import NeighborsInfo


# Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def compute_poly_features(sample_params, recon_image, update_idxs, neighbors: NeighborsInfo,
                          feat_dist_cutoff: float, feat_type: str):
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
    if feat_type == 'polynomial':
        return _compute_polynomial_features(feature)
    elif feat_type == 'rbf':
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
