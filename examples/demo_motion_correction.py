__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'


import nibabel as nib
import os

from niftypad.kt import *
from niftypad.tac import TAC, Ref
from niftypad import basis
from niftypad.image_process.motion_correction import kinetic_model_motion_correction_file



# dt
dt = np.array([[0,15,20,25,30,40,50,60,90,120,180,240,300,450,600,900,1200,1800,2400,3000,3600,4200,4800],
               [15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 3600, 4200,
                4800, 5400]])

# ref
cer_GM = np.array([-18.6264000000000,-106.804500000000,-84.5755000000000,713.409200000000,5286.61550000000,9277.48020000000,11306.7860000000,13052.5925000000,13737.5779000000,13615.7228000000,12984.5333000000,12246.1136000000,11086.0916000000,9520.87550000000,7919.18550000000,6284.80930000000,4824.96600000000,3682.71560000000,3117.62290000000,2733.64120000000,2462.55570000000,2266.43630000000,2126.98140000000])
ref = Ref(cer_GM, dt)
ref.interp_1cubic()

# model
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=[0.0100000/60, 0.300000/60], n_beta=40, w=None,  k2p=0.0501677/60)
model = 'srtmb_basis'
km_inputs = {'b': b}
km_outputs = ['R1', 'k2', 'BP']

# img
file_path = '/Users/Himiko/data/amsterdam_data_pib/transfer_162391_files_89f2cba1/'
pet_file = 'p01_scan1_PET_e2a_128_63_N.hdr'

# frames setup
frame_index = np.arange(dt.shape[-1])
initial_fit_index = frame_index * 1
motion_correction_index = frame_index[20:]
b_initial = basis.make_basis(ref.inputf1cubic, dt[:, initial_fit_index], beta_lim=[0.0100000/60, 0.300000/60], n_beta=40, w=None,  k2p=0.0501677/60)
km_inputs_initial = {'b': b_initial}

# motion initialisation
translation_t = []
rotation_t = []
for t in frame_index:
    translation_t.append([0, 0, 0])
    rotation_t.append([0, 0, 0])

# set iteration
n_iteration = 2
kinetic_model_motion_correction_file(file_path+pet_file, dt, model, km_inputs_initial, km_inputs, km_outputs,
                                     initial_fit_index, motion_correction_index, n_iteration,
                                     translation_t, rotation_t, save_file=True)