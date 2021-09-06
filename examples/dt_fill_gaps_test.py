import matplotlib.pyplot as plt
import numpy as np

from niftypad import basis
from niftypad.kt import dt2mft, dt_fill_gaps
from niftypad.models import get_model_inputs
from niftypad.tac import TAC, Ref

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

beta_lim = [0.0100000 / 60, 0.300000 / 60]
n_beta = 40
k2p = 0.00025
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=None, k2p=k2p)

dt_fill_gaps = dt_fill_gaps(dt)
b_fill_gaps = basis.make_basis(ref.inputf1cubic, dt_fill_gaps, beta_lim=beta_lim, n_beta=n_beta,
                               w=None, k2p=k2p)
inputf1_dt_fill_gaps = ref.inputf1cubic, dt_fill_gaps

models = ['srtmb_basis', 'srtm', 'srtmb_k2p_basis']

user_inputs = {
    'dt': dt, 'inputf1': ref.inputf1cubic, 'w': None, 'r1': 0.905, 'k2p': 0.00025,
    'beta_lim': beta_lim, 'n_beta': n_beta, 'b': b}
user_inputs_fill_gaps = {'b': b_fill_gaps, 'inputf1_dt': inputf1_dt_fill_gaps}
km_outputs = ['R1', 'k2', 'BP']

tac = TAC(ref.tac, dt)
for model_name in models:
    model_inputs = get_model_inputs(user_inputs, model_name)
    tac.run_model(model_name, model_inputs)
    # make
    tac.km_results.update(user_inputs_fill_gaps)
    km_inputs = tac.km_results
    model_km_inputs = get_model_inputs(km_inputs, model_name + '_para2tac')
    tac.run_model_para2tac(model_name + '_para2tac', model_km_inputs)

    plt.plot(tac.mft, tac.tac, 'b*', label='tac')
    plt.plot(dt2mft(dt_fill_gaps), tac.km_results['tacf'], 'r', label='fit')
    plt.legend()
    plt.show()
