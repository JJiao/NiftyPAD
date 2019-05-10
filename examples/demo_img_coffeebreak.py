__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import numpy as np
import nibabel as nib
from niftypad.kt import *
from niftypad.tac import TAC, Ref
from niftypad import basis

# dt
dt = np.array([[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 5396,
                5696, 5996, 6296],
               [5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 3600,
                5696, 5996, 6296, 6596]])

# ref
cer_right_GM = np.array([0, 0, 0, 0, 592.221900000000, 2487.12200000000, 4458.12800000000, 5239.16900000000, 5655.47800000000, 6740.88200000000, 7361.56300000000, 7315.28400000000, 7499.59700000000, 7067.78900000000, 6663.79200000000, 5921.15200000000, 5184.79900000000, 4268.11900000000, 3431.98000000000, 2886.08500000000, 2421.19200000000, 1687.55500000000, 1538.81800000000, 1440.42100000000, 1439.46900000000])
cer_left_GM = np.array([0, 0, 0, 0, 915.895900000000, 3751.55300000000, 5377.27800000000, 5896.48700000000, 6752.62900000000, 7299.80200000000, 7566.03600000000, 7440.07100000000, 7539.30500000000, 7271.21300000000, 6646.04300000000, 6109.30200000000, 5246.95700000000, 4447.90400000000, 3464.82700000000, 2863.01500000000, 2445.84400000000, 1658.95300000000, 1525.59300000000, 1382.73300000000, 1363.93900000000])
cer_GM = cer_right_GM/2 + cer_left_GM/2
ref = Ref(cer_GM, dt)
ref.interp_1cubic()
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=[0.0100000/60, 0.300000/60], n_beta=40, w=None)

# img
filepath = ''
petfile = filepath + 'E301_FLUT_AC1_combined_ID_cleared.nii'
img = nib.load(petfile)
img_data = img.get_data()
img_data[:, :, :, 0:4] = 0
img_r1 = np.zeros(img.shape[0:3])
img_bp = np.zeros(img.shape[0:3])
img_k2 = np.zeros(img.shape[0:3])
thr = 0.001*np.amax(img_data)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            if np.mean(img_data[i, j, k, ]) > thr:
                tac = TAC(img_data[i, j, k, ], dt)
                tac.run_srtmb_basis(b)
                img_r1[i, j, k] = tac.r1
                img_bp[i, j, k] = tac.bp
                img_k2[i, j, k] = tac.k2
save_names = ['r1.nii', 'k2.nii', 'bp.nii']
save_data = [img_r1, img_k2, img_bp]

for i in range(len(save_names)):
    save_data[i] = np.flip(np.flip(save_data[i], axis=1), axis=-1)
    nib.save(nib.Nifti1Image(save_data[i], img.affine), filepath+save_names[i])