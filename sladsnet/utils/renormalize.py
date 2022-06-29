import numpy as np
import tifffile as tif


def renormalize(image: np.ndarray = None, image_path: str = None, max_value: float = 100.0):
    assert (image is not None) or (image_path is not None)
    image_path = image_path
    if image is not None:
        image = image
    else:
        image = tif.imread(image_path)

    image_norm = (image - image.min()) / (image.max() - image.min()) * max_value
    return image_norm


