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
