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

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import tifffile as tif

SUPPORTED_IMAGE_FORMATS = ["tif", "tiff", "npy"]


def renormalize(image: npt.NDArray, max_value: float = 100.0):
    image_norm = (image - image.min()) / (image.max() - image.min()) * max_value
    return image_norm


def load_image_list_renormalize(
    imgs_list: list[Union[str, Path]],
    img_format: str,
    renormalize_images: bool = True,
    max_normalized_value: float = 100,
):
    assert img_format in SUPPORTED_IMAGE_FORMATS

    match img_format:
        case "npy":
            images = [np.load(img) for img in imgs_list]
        case "tif" | "tiff":
            images = [tif.imread(img) for img in imgs_list]

    if renormalize_images:
        images = [renormalize(img, max_value=max_normalized_value) for img in images]
    return images


def load_image_path_renormalize(
    imgs_path: str,
    img_format: str,
    img_suffix: str = None,
    renormalize_images: bool = True,
    max_normalized_value: float = 100,
):

    if img_suffix is None:
        img_suffix = img_format
    image_files = list(Path(imgs_path).glob(f"*.{img_suffix}"))
    return load_image_renormalize(image_files, img_format, renormalize_images, max_normalized_value)
