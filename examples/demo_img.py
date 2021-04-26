__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import os

import nibabel as nib
import numpy as np

from niftypad import basis
from niftypad.image_process.parametric_image import image_to_parametric
from niftypad.models import get_model_inputs
from niftypad.tac import Ref

# dt
dt = np.array([[
    0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400,
    3000, 5396, 5696, 5996, 6296],
               [
                   5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200,
                   1800, 2400, 3000, 3600, 5696, 5996, 6296, 6596]])

# ref
cer_right_GM = np.array([
    0, 0, 0, 0, 592.221900000000, 2487.12200000000, 4458.12800000000, 5239.16900000000,
    5655.47800000000, 6740.88200000000, 7361.56300000000, 7315.28400000000, 7499.59700000000,
    7067.78900000000, 6663.79200000000, 5921.15200000000, 5184.79900000000, 4268.11900000000,
    3431.98000000000, 2886.08500000000, 2421.19200000000, 1687.55500000000, 1538.81800000000,
    1440.42100000000, 1439.46900000000])
cer_left_GM = np.array([
    0, 0, 0, 0, 915.895900000000, 3751.55300000000, 5377.27800000000, 5896.48700000000,
    6752.62900000000, 7299.80200000000, 7566.03600000000, 7440.07100000000, 7539.30500000000,
    7271.21300000000, 6646.04300000000, 6109.30200000000, 5246.95700000000, 4447.90400000000,
    3464.82700000000, 2863.01500000000, 2445.84400000000, 1658.95300000000, 1525.59300000000,
    1382.73300000000, 1363.93900000000])
cer_GM = cer_right_GM/2 + cer_left_GM/2
ref = Ref(cer_GM, dt)
ref.interp_1cubic()

# img file
file_path = ''
pet_file = file_path + 'E301_FLUT_AC1_combined_ID_cleared.nii'
img = nib.load(pet_file)
pet_image = img.get_data()

# k2p
k2p = 0.000250

# basis functions
beta_lim = [0.0100000 / 60, 0.300000 / 60]
n_beta = 40
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=None, k2p=k2p)

# provide all user inputs in one dict here and later 'get_model_inputs' will select the needed ones
user_inputs = {
    'dt': dt, 'inputf1': ref.inputf1cubic, 'w': None, 'r1': 0.905, 'k2p': k2p,
    'beta_lim': beta_lim, 'n_beta': n_beta, 'b': b, 'linear_phase_start': 500,
    'linear_phase_end': None, 'fig': False}

# model
models = [
    'srtmb_basis', 'srtmb_k2p_basis', 'srtmb_asl_basis', 'logan_ref', 'logan_ref_k2p', 'mrtm',
    'mrtm_k2p']
km_outputs = ['R1', 'k2', 'BP']

for model_name in models:
    print(model_name)
    model_inputs = get_model_inputs(user_inputs, model_name)
    parametric_images_dict, pet_image_fit = image_to_parametric(pet_image, dt, model_name,
                                                                model_inputs, km_outputs, thr=0.1)
    for kp in parametric_images_dict.keys():
        nib.save(
            nib.Nifti1Image(parametric_images_dict[kp], img.affine),
            os.path.splitext(img.get_filename())[0] + '_' + model_name + '_' + kp +
            os.path.splitext(img.get_filename())[1])
    nib.save(
        nib.Nifti1Image(pet_image_fit, img.affine),
        os.path.splitext(img.get_filename())[0] + '_' + model_name + '_' + 'fit' +
        os.path.splitext(img.get_filename())[1])
