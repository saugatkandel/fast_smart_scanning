import dataclasses as dt

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor as nnr

from ..input_params import SladsModelParams


class ERDModel:
    def __init__(self, model_type: str):
        if model_type != 'slads-net':
            raise NotImplementedError
        self.model_type = model_type
        self._fitted = False
        self.model = None

    def predict(self, poly_features):
        assert self._fitted
        return self.model.predict(poly_features)


class SladsSklearnModel(ERDModel):
    def __init__(self, load_path: str = None,
                 model_params: SladsModelParams = None):
        super().__init__(model_type='slads-net')

        if load_path is not None:
            self._load_from_path(load_path)
            self._fitted = True
        else:
            self._setup_training(model_params)
            self.model_path = None
            self._fitted = False

    def _load_from_path(self, model_path: str):
        self.model_params = None
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def _setup_training(self, model_params):
        self.model_params = SladsModelParams() if model_params is None else model_params
        self.model = nnr(**dt.asdict(self.model_params))

    def fit(self, poly_features: np.ndarray, erds: np.ndarray):
        self.model.fit(poly_features, erds)
        self._fitted = True

    def save(self, save_path):
        assert self._fitted
        self.model_path = save_path
        joblib.dump(self.model, save_path)
