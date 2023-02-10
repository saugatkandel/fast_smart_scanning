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
from pathlib import Path

import joblib

from .base import Sample


@dt.dataclass
class ResultOutputs:
    poly_features: list
    ratio_measured: list
    erds: list
    masks: list
    recons: list


class Result:
    def __init__(self, c_value: float = None):
        self.c_value = c_value
        self.outputs = ResultOutputs([], [], [], [], [])
        self.size = 0

    def add(self, sample: Sample):
        self.outputs.poly_features.append(sample.poly_features.copy().astype("float32"))
        self.outputs.erds.append(sample.ERD.copy().astype("float32"))
        self.outputs.ratio_measured.append(sample.ratio_measured.astype("float32"))
        self.outputs.masks.append(sample.mask.copy().astype("bool"))
        self.outputs.recons.append(sample.recon_image.copy().astype("float32"))
        self.size += 1

    def get_by_index(self, index):
        out_dict = {f.name: getattr(self.outputs, f.name)[index] for f in dt.fields(self.outputs)}
        return out_dict

    def get(self):
        return dt.asdict(self.outputs)

    def save(self, path: str):
        path = Path(path)
        if path.suffix != ".pkl":
            path = Path(f"{path}.pkl")
        joblib.dump(self.get(), path)
