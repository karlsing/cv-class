import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """
    Computes the mean and the standard deviation of all images present within
    the directory.

    Note: convert the image in grayscale and then in [0,1] before computing the
    mean and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = 1 / Variance

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None
    ############################################################################
    # Student code begin
    ############################################################################

    filenames = glob.glob(os.path.join(dir_name,"**/*.jpg"), recursive=True)
    images = map(lambda x:np.asarray(Image.open(x).convert('L'))/255.0, filenames)
    count_number = 0
    all_sum = 0
    all_sum_sq = 0
    for image in images:
        pixel_sum = np.sum(image)
        pixel_sum_square = np.sum(np.square(image))
        count_number += image.size
        all_sum += pixel_sum
        all_sum_sq += pixel_sum_square
    mean = all_sum / count_number
    std = np.sqrt((1 / (count_number - 1)) * (all_sum_sq - count_number * (mean ** 2)))

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
