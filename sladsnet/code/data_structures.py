#! /usr/bin/env python3
import dataclasses as dt

import numpy as np


@dt.dataclass
class NeighborsInfo:
    indices: list = dt.field(default_factory=lambda: [])
    distances: list = dt.field(default_factory=lambda: [])
    weights: list = dt.field(default_factory=lambda: [])
    values: list = dt.field(default_factory=lambda: [])


@dt.dataclass
class MeasurementInfo:
    measured_idxs: np.ndarray = dt.field(default_factory=lambda: np.empty((0, 2), dtype='int'))
    unmeasured_idxs: np.ndarray = dt.field(default_factory=lambda: np.empty((0, 2), dtype='int'))
    measured_values: np.ndarray = dt.field(default_factory=lambda: np.empty(0, dtype='float32'))
    unnormalized_values: np.ndarray = dt.field(default_factory=lambda: np.empty(0, dtype='float32'))
    new_idxs: list = dt.field(default_factory=lambda: [])
