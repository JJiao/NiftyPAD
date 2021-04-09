__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import csv
import os

import nibabel as nib

from .parametric_image import image_to_parametric, parametric_to_image
from .transforms import image_registration_3d_plus_t


def kinetic_model_motion_correction(pet_image, dt, model, km_inputs_initial, km_inputs, km_outputs,
                                    initial_fit_index, motion_correction_index, n_iteration,
                                    translation_t, rotation_t):
    parametric_images_dict, _ = image_to_parametric(pet_image[:, :, :, initial_fit_index],
                                                    dt[:, initial_fit_index], model,
                                                    km_inputs_initial, km_outputs)
    pet_image_fit = parametric_to_image(parametric_images_dict, dt, model, km_inputs)
    for ii in range(n_iteration):
        print('iteration ' + str(ii + 1))
        print('updating motion ...')
        translation_t, rotation_t, pet_image_realigned = image_registration_3d_plus_t(
            pet_image, pet_image_fit, translation_t, rotation_t, motion_correction_index)
        print('updating kinetics ...')
        parametric_images_dict, pet_image_fit = image_to_parametric(pet_image_realigned, dt, model,
                                                                    km_inputs, km_outputs)
    return pet_image_realigned, translation_t, rotation_t, parametric_images_dict


def kinetic_model_motion_correction_file(pet_file, dt, model, km_inputs_initial, km_inputs,
                                         km_outputs, initial_fit_index, motion_correction_index,
                                         n_iteration, translation_t, rotation_t, save_file):
    pet_img = nib.load(pet_file)
    pet_img_data = pet_img.get_data()
    # pet_img_data[pet_img_data < 0] = 0
    pet_img_data[:, :, 0:1, :] = 0
    pet_image_realigned, translation_t, rotation_t, parametric_images_dict = \
        kinetic_model_motion_correction(
            pet_img_data, dt, model, km_inputs_initial, km_inputs, km_outputs, initial_fit_index,
            motion_correction_index, n_iteration, translation_t, rotation_t)
    if save_file:
        parametric_images_dict.update({'mc': pet_image_realigned})
        for kk in parametric_images_dict.keys():
            nib.save(nib.Nifti1Image(parametric_images_dict[kk].squeeze(), pet_img.affine),
                     os.path.splitext(pet_file)[0] + '_mc_' + model + '_' + kk + '.nii')
        motion_file_name = os.path.splitext(pet_file)[0] + '_' + model + '_motion' + '.csv'
        motion_file = open(motion_file_name, 'w')
        motion_writer = csv.writer(motion_file, dialect='excel-tab', delimiter=',')
        motion_writer.writerows(list(zip(translation_t, rotation_t)))
        motion_file.close()
