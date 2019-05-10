__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np


def extract_regional_values(image, parcellation, labels):
    idx = labels_to_index(parcellation, labels)
    regional_values = np.mean(image[idx, ], axis=0)
    return regional_values


def labels_to_index(parcellation, labels):
    parcellation = np.squeeze(parcellation)
    idx = np.zeros(parcellation.shape, dtype='bool')
    for i in range(len(labels)):
        idx = np.logical_or(idx, parcellation == labels[i])
    return idx



