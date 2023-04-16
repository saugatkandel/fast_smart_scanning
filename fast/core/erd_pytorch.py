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
import dataclasses as dt

import joblib
import numpy as np
import torch
from torch import nn

from ..input_params import SladsPytorchModelParams
from .erd import ERDModel

ACTIVATIONS = {"leakyrelu": torch.nn.LeakyReLU, "relu": torch.nn.ReLU, "identity": torch.nn.Identity}


class NeuralNetwork(nn.Module):
    def __init__(self, n_layers: int = 5, n_hidden_layers: int = 50, activation: str = "leakyrelu"):
        super(NeuralNetwork, self).__init__()

        activation_fn = ACTIVATIONS[activation]

        stack = []
        for n in range(n_layers):
            if n != n_layers - 1:
                stack.append(nn.LazyLinear(n_hidden_layers))
            else:
                stack.append(nn.LazyLinear(1))
            stack.append(activation_fn())

        self.linear_stack = nn.Sequential(*stack)

    def forward(self, x):
        erds = self.linear_stack(x)
        return erds


class SladsPytorchModel(ERDModel):
    def __init__(self, load_path: str = None, model_params: SladsPytorchModelParams = None):
        super().__init__(model_type="slads-net-pytorch")
        self.model_params = model_params

        if load_path is not None:
            self._load_from_path(load_path)
            self._fitted = True
        else:
            self._setup_training(model_params)
            self.model_path = None
            self._fitted = False

    def _load_from_path(self, model_path: str):
        self.model_path = model_path

        model_state_dict = torch.load(self.model_path)
        cuda_status = torch.backends.cuda.isavailable()
        mps_status = torch.backends

    def _setup_training(self, model_params):
        self.model_params = SladsPytorchModelParams() if model_params is None else model_params

        self.model = nnr(**dt.asdict(self.model_params))

    def fit(self, poly_features: np.ndarray, erds: np.ndarray):
        self.model.fit(poly_features, erds)
        self._fitted = True

    def save(self, save_path):
        assert self._fitted
        self.model_path = save_path
        joblib.dump(self.model, save_path)
