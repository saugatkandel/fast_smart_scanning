import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data_structures import NeighborsInfo, MeasurementInfo


def find_neighbors(minfo: MeasurementInfo, num_neighbors: int, resolution: int = 1):
    neigh = NearestNeighbors(n_neighbors=num_neighbors)
    neigh.fit(minfo.measured_idxs)
    neighbor_distances, neighbor_indices = neigh.kneighbors(minfo.unmeasured_idxs)
    neighbor_distances = neighbor_distances * resolution
    neighbor_values = minfo.measured_values[neighbor_indices]
    neighbor_weights = _compute_neighbor_weights(neighbor_distances)

    return NeighborsInfo(indices=neighbor_indices, distances=neighbor_distances, weights=neighbor_weights,
                         values=neighbor_values)


def _compute_neighbor_weights(neighbor_distances, power=2):
    """Calculating the weights for how each neighboring data point contributes
    to the reconstruction for the current location.

    First, the weights are calculated to be inversely proportional to the distance from teh current point.
    Next, the weights are normalized so that the total weight sums up to 1 for each reconstruction point."""

    unnormalized_weights = 1 / np.power(neighbor_distances, power)
    sum_over_row = np.sum(unnormalized_weights, axis=1, keepdims=True)
    weights = unnormalized_weights / sum_over_row
    return weights
