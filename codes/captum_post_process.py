import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from enum import Enum

from scipy.stats import truncnorm
import torch

'''
Procedure: 

1. Truncate the image [0, 255]
2. Normalize Attribution requires the following steps:
    A. Collapse all channels by summing them up. Let's call the result S. 
    B. Calculate cut-off threshold value up to which sum of attributions is e.g. 98% of total attributions. Let's call the value V
    C. Normalize S by the value V. Let's call it norm_attr
    D. Return norm_attr
'''


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

def _prepare_image(attr_visual):
    return np.clip(attr_visual.astype(int), 0, 255)

def _normalize_scale(attr, scale_factor):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values, percentile):
    '''
    Find the max value at which cumulative sum covers up the 98 percentile of the total attributions
    '''
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    # Mark the value up to which sum of attributions is e.g. 98% of total attributions
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    
    return sorted_vals[threshold_id]

def _normalize_image_attr(
    attr, sign, outlier_perc = 2
):
    attr_combined = np.sum(attr, axis=2)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")

    return _normalize_scale(attr_combined, threshold)


if __name__ == "__main__":
    
    original_image = np.random.uniform(0, 1.0, (4,4,3))

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = _prepare_image(original_image * 255)

    print(np.max(original_image))

    sign = 'absolute_value'
    s = np.random.normal(0, 0.001, (4, 4, 3))
    print(s)
    print(s.shape)
    dummy = np.sum(s, axis=2)
    print(dummy/np.max(np.abs(dummy)))
    norm_attr = _normalize_image_attr(s, sign)
    print(norm_attr)

    # masked_img will be highly required
    mask_img= _prepare_image(original_image * np.expand_dims(norm_attr, 2))

    print(mask_img)
    print(mask_img.shape)

# alpha_scaled = np.concatenate(
#                     [
#                         original_image,
#                         _prepare_image(np.expand_dims(norm_attr, 2) * 255),
#                     ],
#                     axis=2,
#                 )
#
# print(alpha_scaled.shape)