import dataclasses as dt

import numpy as np

SUPPORTED_FEATURE_TYPES = ['polynomial', 'rbf']
SUPPORTED_MODEL_TYPES = ['slads-net']
SUPPORTED_SAMPLING_TYPES = ['slow_detailed', 'fast_limited']
SCAN_TYPES = ['transmission', 'diffraction']
GRID_TO_PIXEL_METHODS = ['pixel', 'subpixel']
SIMULATION_TYPES = ['training', 'emulation', 'visualize_erd']


@dt.dataclass(frozen=True)
class TrainingInputParams:
    input_images_path: str
    output_dir: str
    initial_scan_ratio: float = 0.01 # initial scan ratio
    stop_ratio: float = 0.8 # stop ratio
    scan_method: str = 'random' # pointwise or line scan
    scan_type: str = 'transmission'
    sampling_type: str = 'fast_limited'
    num_repeats_per_mask: int = 2
    measurements_per_initial_mask: int = 50 # Only used if sampling type is fast_limited
    random_seed: int = 111
    training_split: float = 0.9
    test_c_values: list = dt.field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    calculate_full_erd_per_step: bool = True

    def __post_init__(self):
        assert self.sampling_type in SUPPORTED_SAMPLING_TYPES


@dt.dataclass(frozen=True)
class SladsModelParams:
    random_state: int = 1
    activation: str = 'relu'
    hidden_layer_sizes: tuple = (50, 5)
    solver: str = 'adam'
    max_iter: int = 500
    alpha: float = 1e-4


@dt.dataclass(frozen=True)
class ERDInputParams:
    c_value: float
    model_type: str = 'slads-net'
    static_window: bool = False
    static_window_size: int = 15
    dynamic_window_sigma_mult: float = 3
    feat_distance_cutoff: float = 0.25
    feature_type: str = 'rbf'
    calculate_full_erd_per_step: bool = False
    full_erd_recalculation_frequency: int = 10
    affected_neighbors_window_min: float = 10
    affected_neighbors_window_max: float = 20
    affected_window_increase_factor: float = 1.5

    def __post_init__(self):
        assert self.model_type in SUPPORTED_MODEL_TYPES
        assert self.feature_type in SUPPORTED_FEATURE_TYPES


@dt.dataclass(frozen=True)
class GeneralInputParams:
    desired_td: float = 0
    num_neighbors: int = 10
    neighbor_weight_distance_norm: float = 2.0


@dt.dataclass
class SampleParams:
    """Contains information that is held constant throughout the procedure."""
    image_shape: list # tuple containing image shape
    initial_scan_ratio: float = 0.01# initial scan ratio
    stop_ratio: float = 0.4 # stop ratio
    scan_method: str ='pointwise'# pointwise or random scan
    scan_type: str = 'transmission'
    outer_batch_size: int = 1
    inner_batch_size: int = 1
    random_seed: int = None
    initial_idxs: list = None
    initial_mask: np.ndarray = dt.field(init=False)
    line_revisit: bool = False # not in use right now
    rng: np.random.Generator = None
    image_size: int = dt.field(init=False)

    def __post_init__(self):
        if self.rng is not None:
            if self.random_seed is not None:
                raise ValueError('Cannot supply both rng and random seed.')
        else:
            self.rng = np.random.default_rng(self.random_seed)
        if self.initial_idxs is None:
            self.initial_idxs, self.initial_mask = self.generate_initial_mask(self.scan_method)
        else:
            self.initial_mask = np.zeros(self.image_shape)
            self.initial_mask[self.initial_idxs[:, 0], self.initial_idxs[:, 1]] = 1

        self.image_size = self.image_shape[0] * self.image_shape[1]

        if self.scan_type != 'transmission':
            raise NotImplementedError

    def generate_initial_mask(self, scan_method):
        # Update the scan method
        self.scan_method = scan_method

        # List of what points/lines should be initially measured

        # If scanning with line-bounded constraint
        if self.scan_method == 'linewise':
            new_idxs, new_mask = self._gen_linewise_scan()
        elif self.scan_method == 'pointwise' or self.scan_method == 'random':
            new_idxs, new_mask = self._gen_pointwise_scan()
        else:
            raise NotImplementedError
        return new_idxs, new_mask

    def _gen_pointwise_scan(self):
        # Randomly select points to initially scan
        mask = self.rng.random(self.image_shape)
        mask = (mask <= self.initial_scan_ratio)
        new_idxs = np.array(np.where(mask)).T
        return new_idxs, mask

    def _gen_linewise_scan(self):
        raise NotImplementedError

        # Create list of arrays containing points to measure on each line
        self.linesToScan = np.asarray(
            [[tuple([rowNum, columnNum]) for columnNum in np.arange(0, self.finalDim[1], 1)] for rowNum in
             np.arange(0, self.finalDim[0], 1)]).tolist()

        # Set initial lines to scan
        lineIndexes = [int(round((self.finalDim[0] - 1) * startLinePosition)) for startLinePosition in
                       startLinePositions]

        # Obtain points in the specified lines and add them to the initial scan list
        for lineIndex in lineIndexes:

            # If only a percentage should be scanned, then randomly select points, otherwise select all
            if lineMethod == 'percLine':
                newIdxs = copy.deepcopy(self.linesToScan[lineIndex])
                np.random.shuffle(newIdxs)
                newIdxs = newIdxs[:int(np.ceil((self.stopPerc / 100) * self.finalDim[1]))]
            else:
                newIdxs = [pt for pt in self.linesToScan[lineIndex]]

            # Add positions to initial list
            self.initialSets.append(newIdxs)


class SimulatedSampleParams(SampleParams):
    def __init__(self, image, simulation_type, *args, **kwargs):
        assert simulation_type in SIMULATION_TYPES
        self.image = image
        self.simulation_type = simulation_type
        super().__init__(image.shape, *args, **kwargs)