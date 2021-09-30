__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import numpy as np
from tqdm import trange

from ..kt import dt2tdur
from ..tac import TAC
from .regions import extract_regional_values


def image_to_parametric(pet_image, dt, model_name, model_inputs, km_outputs, thr=0.005):
    parametric_images = []
    tac = TAC(pet_image[0, 0, 0,] * 0 + 1, dt)
    tac.run_model(model_name, model_inputs)
    km_outputs = list({c.lower() for c in km_outputs} & set(tac.km_results.keys()))
    for _ in range(len(km_outputs)):
        parametric_images.append(np.zeros(pet_image.shape[0:3]))
    pet_image_fit = np.zeros(pet_image.shape)
    thr = thr * np.amax(pet_image)
    mask = np.argwhere(np.mean(pet_image, axis=-1) > thr)
    for i in trange(mask.shape[0]):
        tac = TAC(pet_image[mask[i][0], mask[i][1], mask[i][2],], dt)
        tac.run_model(model_name, model_inputs)
        # # #
        # plt.plot(tac.mft, tac.km_results['tacf'],'r', tac.mft, tac.tac, 'go')
        # plt.show()
        # # #
        if 'tacf' in tac.km_results:
            pet_image_fit[mask[i][0], mask[i][1], mask[i][2],] = tac.km_results['tacf']
        # # #
        # print(tac.km_results)
        # # #
        for p in range(len(km_outputs)):
            parametric_images[p][mask[i][0], mask[i][1],
                                 mask[i][2],] = tac.km_results[km_outputs[p].lower()]
    parametric_images_dict = dict(zip(km_outputs, parametric_images))
    if 'tacf' in tac.km_results:
        return parametric_images_dict, pet_image_fit
    return parametric_images_dict


def parametric_to_image(parametric_images_dict, dt, model, km_inputs):
    parametric_images = list(parametric_images_dict.values())
    pet_image = np.zeros(parametric_images[0].shape[0:3] + (dt.shape[-1],))
    mask = np.argwhere(parametric_images[0] != 0)
    for i in range(mask.shape[0]):
        km_inputs_local = km_inputs.copy()
        for p in parametric_images_dict.keys():
            km_inputs_local.update({
                p.lower(): parametric_images_dict[p][mask[i][0], mask[i][1], mask[i][2]]})
        tac = TAC([], dt)
        getattr(tac, 'run_' + model + '_para2tac')(**km_inputs_local)
        pet_image[mask[i][0], mask[i][1], mask[i][2],] = tac.km_results['tacf']
    return pet_image


def image_to_suvr_with_parcellation(pet_image, dt, parcellation, reference_region_labels,
                                    selected_frame_index):
    reference_regional_tac = extract_regional_values(pet_image, parcellation,
                                                     reference_region_labels)
    image_suvr = image_to_suvr_with_reference_tac(pet_image, dt, reference_regional_tac,
                                                  selected_frame_index)
    return image_suvr


def image_to_suvr_with_reference_tac(pet_image, dt, reference_regional_tac, selected_frame_index):
    tdur = dt2tdur(dt)
    image = np.multiply(pet_image[:, :, :, selected_frame_index], tdur[selected_frame_index])
    ref = np.multiply(reference_regional_tac[selected_frame_index], tdur[selected_frame_index])
    image_suvr = np.sum(image, axis=-1) / np.sum(ref)
    return image_suvr
