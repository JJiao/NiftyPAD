__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import numpy as np

from niftypad import basis
from niftypad.kt import dt2tdur
from niftypad.models import get_model_inputs
from niftypad.tac import TAC, Ref

# test data
dt = np.array([[
    0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360, 420, 480, 540, 600, 660, 720,
    780, 840, 1020, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300],
               [
                   15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360, 420, 480, 540, 600,
                   660, 720, 780, 840, 1020, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300,
                   3600]])

ref = np.array([
    0.125056713393401, 2.72027858709524, 14.6002097680789, 19.2674706945717, 22.0480095571301,
    23.9441317503439, 24.2691083273475, 23.9083322213152, 23.7586098409017, 23.2083883936197,
    22.5765266094252, 22.1952340572856, 21.2696724932646, 20.0587051261558, 19.0773912540726,
    17.9360174963354, 17.0221288817712, 16.3031396647599, 15.2162922022706, 14.5955039576435,
    13.8676580563113, 12.4044758413027, 10.7075315937501, 8.85178441550057, 7.29047317769810,
    6.22241739146864, 5.38418292132310, 4.82308160452273, 4.51763401957080, 4.23271724360837,
    3.99675706316514])

tac = np.array([
    0.0612937319754616, 2.56733416202755, 13.2302628562565, 17.5212376697891, 19.8554469827973,
    21.2888518029975, 22.3164453083215, 21.9106339934451, 21.6976991361882, 21.2760967663021,
    21.0320027737135, 20.3356078483447, 19.7789261620624, 18.7016942272702, 17.5381378584246,
    16.6955628809652, 15.8305514461795, 15.1466577726934, 14.2137260024906, 13.7598553636470,
    12.9989332998008, 11.8293518915139, 10.3648913297944, 8.93110113265639, 7.50503727314817,
    6.49229487908344, 5.83027369667342, 5.35683342894270, 4.94744092837665, 4.79454967790696,
    4.49907835991146])

tac = TAC(tac, dt)
ref = Ref(ref, dt)
ref.interp_1()
w = dt2tdur(tac.dt)
w = w / np.amax(w)
beta_lim = [0.0100000 / 60, 0.300000 / 60]
n_beta = 40
b = basis.make_basis(ref.inputf1, dt, beta_lim=beta_lim, n_beta=n_beta, w=w)

# provide all user inputs in one dict here and later 'get_model_inputs' will select the needed ones
user_inputs = {
    'dt': dt, 'inputf1': ref.inputf1, 'w': w, 'r1': 0.905, 'k2p': 0.00025, 'beta_lim': beta_lim,
    'n_beta': n_beta, 'b': b, 'linear_phase_start': 500, 'linear_phase_end': None, 'fig': True}

models = [
    'srtm', 'srtmb', 'srtmb_basis', 'srtmb_asl', 'srtmb_k2p', 'logan_ref', 'logan_ref_k2p', 'mrtm',
    'mrtm_k2p']

for model_name in models:
    model_inputs = get_model_inputs(user_inputs, model_name)
    tac.run_model(model_name, model_inputs)
    print(tac.km_results)
