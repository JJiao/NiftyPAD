__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np


def srtm_theta2kp(theta):
    # theta - output from models/srtmb
    # theta[0] - linear coeff of input(Cr)
    # theta[1] - linear coeff of the conv
    # theta[2] - exp coeff (no -)

    r1 = theta[0]                       # R1
    k2 = theta[1] + theta[0] * theta[2] # k2
    bp = k2 / theta[2] - 1              # BPnd
    return r1, k2, bp


def srtm_kp2theta(r1, k2, bp):
    theta = np.zeros(3)
    theta[0] = r1
    theta[1] = k2 * (1 - r1 / (1+bp))
    theta[2] = k2 / (1+bp)
    return theta
