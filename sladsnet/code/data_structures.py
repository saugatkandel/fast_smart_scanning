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
    measured_idxs: np.ndarray
    unmeasured_idxs: np.ndarray
    measured_values: np.ndarray
    new_idxs: list = dt.field(default_factory=lambda: [])
