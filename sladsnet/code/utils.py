import numpy as np
import tifffile as tif


def renormalize_to_grayscale(image: np.ndarray = None, image_path: str = None):
    assert (image is not None) or (image_path is not None)
    image_path = image_path
    if image is not None:
        image = image
    else:
        image = tif.imread(image_path)

    image_norm = (image - image.min()) / (image.max() - image.min()) * 255
    return image_norm


