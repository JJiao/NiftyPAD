__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

# weighting scheme implementation based on https://www.ncbi.nlm.nih.gov/pubmed/16912378

import numpy as np

from . import kt


def get_weight(dt, isotope='18F', a=1, t=None, r=None, tac=None):

    if isotope == '18F':
        half_life = 20.39 * 60
    elif isotope == '12C':
        half_life = 109.77 * 60
    else:
        print('not known yet')

    decay_const = np.log(2) / half_life
    tdur = kt.dt2tdur(dt)
    dcf = decay_const * tdur / (np.exp(-decay_const * dt[0,]) - np.exp(-decay_const * dt[1,]))

    n_model = 5
    w_m = [None] * n_model
    i = 0
    w_m[i] = a * dcf * dcf * t / tdur / tdur
    i += 1
    w_m[i] = a * dcf * dcf * (t + 2*r) / tdur / tdur
    i += 1
    w_m[i] = a
    i += 1
    w_m[i] = a * dcf * dcf
    i += 1
    w_m[i] = a * tac * tdur * dcf

    for i in range(n_model):
        w_m[i] = 1 / w_m[i]

    return w_m


def get_weight_ppet(dt, tac, isotope='12C', a=1):
    if isotope == '18F':
        half_life = 20.39 * 60
    elif isotope == '12C':
        half_life = 109.77 * 60
    else:
        print('not known yet')
    decay_const = np.log(2) / half_life

    tdur = kt.dt2tdur(dt)
    dcf = decay_const * tdur / (np.exp(-decay_const * dt[0,]) - np.exp(-decay_const * dt[1,]))
    tac_0 = tac * 1
    tac[tac < 0] = 1

    abundance = 1 / a
    prompts = abundance / dcf * tdur * tac
    normalised = prompts / np.max(prompts) * 100000000
    w = tdur * tdur / normalised / dcf
    w[tac_0 < 0] = 0
    return w
