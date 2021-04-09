__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import nibabel as nib
import numpy as np


def extract_regional_values(image, parcellation, labels):
    idx = labels_to_index(parcellation, labels)
    regional_values = np.mean(image[idx,], axis=0)
    return regional_values


def labels_to_index(parcellation, labels):
    parcellation = np.squeeze(parcellation)
    idx = np.zeros(parcellation.shape, dtype='bool')
    for i in range(len(labels)):
        idx = np.logical_or(idx, parcellation == labels[i])
    return idx


def extract_regional_values_image_file(image_file, parcellation_file):
    image = nib.load(image_file)
    image_data = image.get_data()
    n_frames = 1
    if image_data.ndim == 4:
        n_frames = image_data.shape[-1]
    parcellation_img = nib.load(parcellation_file)
    parcellation = parcellation_img.get_data()
    regions_label = np.unique(parcellation)
    regions_data = np.zeros((regions_label.size, n_frames))
    regions_data = np.squeeze(regions_data)
    for i in range(regions_label.size):
        regions_data[i] = extract_regional_values(image_data, parcellation, [regions_label[i]])
    return regions_data, regions_label
