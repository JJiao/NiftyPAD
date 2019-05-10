__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import nibabel as nib
import csv
import os

from niftypad.tac import TAC, Ref
from niftypad import basis
from niftypad.kt import *
from niftypad.image_proc.regions import extract_regional_values

# dt
dt = np.array([[0,15,20,25,30,40,50,60,90,120,180,240,300,450,600,900,1200,1800,2400,3000,3600,4200,4800],
               [15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 3600, 4200,
                4800, 5400]])

# ref
cer_GM = np.array([-18.6264000000000,-106.804500000000,-84.5755000000000,713.409200000000,5286.61550000000,9277.48020000000,11306.7860000000,13052.5925000000,13737.5779000000,13615.7228000000,12984.5333000000,12246.1136000000,11086.0916000000,9520.87550000000,7919.18550000000,6284.80930000000,4824.96600000000,3682.71560000000,3117.62290000000,2733.64120000000,2462.55570000000,2266.43630000000,2126.98140000000])
ref = Ref(cer_GM, dt)
ref.interp_1cubic()

# models
models = ['srtmb_basis', 'srtmb_k2p_basis']
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=[0.0100000/60, 0.300000/60], n_beta=40, w=None, k2p=0.0501677/60)
km_inputs = {'b': b}
km_outputs = ['R1', 'k2', 'BP']

# img
file_path = '/Users/Himiko/data/amsterdam_data_pib/transfer_162391_files_89f2cba1/'
pet_file = 'p01_scan1_PET_e2a_128_63_N.hdr'
parcellation_file = 'p01_scan1_PET_e2a_128_63_N_GMWM_ROI.hdr'
pet_img = nib.load(file_path + pet_file)
parcellation_img = nib.load(file_path + parcellation_file)
img_data = pet_img.get_data()
parcellation = parcellation_img.get_data().astype(np.int16)
regions_label = np.unique(parcellation)

# write the csv header info
results_file_name = file_path + os.path.splitext(pet_file)[0] + '_' + '_'.join(models) + '.csv'
results_file = open(results_file_name, 'w', newline='')
results_writer = csv.writer(results_file, dialect='excel-tab',delimiter=',')
results_writer.writerow(['label'] + [mm + '_' + kk for mm in models for kk in km_outputs])

# do analysis and write to file

for i in range(regions_label.size):
    region_results = []
    region_results.append(regions_label[i])
    tac_data = extract_regional_values(img_data, parcellation, [regions_label[i]])
    tac = TAC(tac_data, dt)
    for m in range(len(models)):
        getattr(tac, 'run_' + models[m])(**km_inputs)
        for p in range(len(km_outputs)):
            region_results.append(getattr(tac, km_outputs[p].lower()))
    results_writer.writerow(region_results)
results_file.close()



