#! /usr/bin/env python3
import dataclasses as dt

import numpy as np


@dt.dataclass
class NeighborsInfo:
    indices: list
    distances: list
    weights: list
    values: list


@dt.dataclass
class MeasurementInfo:
    measured_idxs: np.ndarray = dt.field(default_factory=lambda: np.empty((0, 2), dtype='int'))
    unmeasured_idxs: np.ndarray = dt.field(default_factory=lambda: np.empty((0, 2), dtype='int'))
    measured_values: np.ndarray = dt.field(default_factory=lambda: np.empty(0, dtype='float32'))
    new_idxs: list = dt.field(default_factory=lambda: [])
