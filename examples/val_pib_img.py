__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import numpy as np
import nibabel as nib
import os
from NiftyPAD.niftypad.kt import *
from NiftyPAD.niftypad.tac import TAC, Ref
from NiftyPAD.niftypad import basis


# dt
dt = np.array([[0,15,20,25,30,40,50,60,90,120,180,240,300,450,600,900,1200,1800,2400,3000,3600,4200,4800],
               [15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 3600, 4200,
                4800, 5400]])

# ref
cer_GM = np.array([-18.6264000000000,-106.804500000000,-84.5755000000000,713.409200000000,5286.61550000000,9277.48020000000,11306.7860000000,13052.5925000000,13737.5779000000,13615.7228000000,12984.5333000000,12246.1136000000,11086.0916000000,9520.87550000000,7919.18550000000,6284.80930000000,4824.96600000000,3682.71560000000,3117.62290000000,2733.64120000000,2462.55570000000,2266.43630000000,2126.98140000000])
ref = Ref(cer_GM, dt)
ref.interp_1cubic()

# models
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=[0.0100000/60, 0.300000/60], n_beta=40, w=None,  k2p=0.0501677/60)
models = ['srtmb_basis', 'srtmb_k2p_basis']
km_inputs = {'b': b}
km_outputs = ['R1', 'k2', 'BP']

# img
file_path = '/Users/Himiko/data/amsterdam_data_pib/transfer_162391_files_89f2cba1/'
pet_file = 'p01_scan1_PET_e2a_128_63_N.hdr'
img = nib.load(file_path + pet_file)
img_data = img.get_data()

# get results
img_results = np.zeros(img.shape[0:3] + (len(km_outputs), len(models)))
thr = 0.005*np.amax(img_data)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            if np.mean(img_data[i, j, k, ]) > thr:
                tac = TAC(img_data[i, j, k, ], dt)
                for m in range(len(models)):
                    getattr(tac, 'run_' + models[m])(**km_inputs)
                    for p in range(len(km_outputs)):
                        img_results[i, j, k, p, m] = getattr(tac, km_outputs[p].lower())


# save results
for m in range(len(models)):
    for p in range(len(km_outputs)):
        nib.save(nib.Nifti1Image(img_results[:, :, :, p, m].squeeze(), img.affine), file_path +
                 os.path.splitext(pet_file)[0] + '_' + models[m] + '_' + km_outputs[p] + '.nii')