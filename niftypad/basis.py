__author__ = 'jieqing jiao'

import numpy as np
from . import kt
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.io import savemat



def make_basis(inputf1, dt, beta_lim=[10e-6, 10e-1], n_beta=128, beta_space='log', w=None, k2p=None, fig=False):

    # dt can skip frames if needed

    # calculate input based on dt
    tdur = kt.dt2tdur(dt)
    # input = kt.int2dt(inputf1, dt)
    t1 = np.arange(np.amax(dt))

    # generate basis functions and m based on dt
    if beta_space == 'log':
        # # PPET beta
        # beta = np.logspace(start=np.log10(beta_lim[0]), stop=np.log10(beta_lim[1]), num=n_beta, endpoint=False)

        beta = np.logspace(start=np.log10(beta_lim[0]), stop=np.log10(beta_lim[1]), num=n_beta, endpoint=True)
        # # QM beta
        # beta = np.logspace(start=np.log10(beta_lim[0]*60), stop=np.log10(beta_lim[1]*60), num=n_beta, endpoint=True)
        # beta /= 60
        # same as
        #         beta = np.logspace(start=np.log10(beta_lim[0]), stop=np.log10(beta_lim[1]), num=n_beta, endpoint=True)
        # print(beta)


        # beta = np.logspace(start=np.log(beta_lim[0]), stop=np.log(beta_lim[1]), num=n_beta, endpoint=True, base=np.exp(1))
    elif beta_space == 'natural':
        beta = np.zeros(n_beta)
        for i in range(0, n_beta):
            beta[i] = - 1 / t1[-1] * \
                      np.log((np.exp(-beta_lim[0]*t1[-1]) - np.exp(-beta_lim[1]*t1[-1])) / n_beta * i +
                             np.exp(-beta_lim[1]*t1[-1]))
    basis = np.zeros((beta.size, tdur.size))
    basis_w = np.zeros((beta.size, tdur.size))
    m = np.zeros((beta.size*2, tdur.size))
    m_w = np.zeros((beta.size*2, tdur.size))
    basis_k2p = np.zeros((beta.size, tdur.size)) # no need to calculate basis_k2p_w

    for i in range(0, beta.size):
        # make basis
        # inputf1[inputf1 < 0] = 0.0
        basis1 = np.convolve(inputf1, np.exp(-beta[i]*t1))

        basis[i, :] = kt.int2dt(basis1, dt)
        input = kt.int2dt(inputf1, dt)
        # print(input)

        # # QM basis and input
        mft = kt.dt2mft(dt)
        basis_f = interp1d(t1, basis1[:t1.size], kind='linear', fill_value='extrapolate')
        basis[i, :] = basis_f(mft)
        inputf = interp1d(np.arange(inputf1.size), inputf1, kind='linear')
        input = inputf(mft)
        # print(input)


        # print(input)
        if w is not None:
            basis_w[i, :] = basis[i, :] * w
        # make m
        a = np.column_stack((input, basis[i, :]))
        q, r = np.linalg.qr(a)
        m[i*2:i*2+2, :] = np.linalg.solve(r, q.T)
        if w is not None:
            a = np.dot(np.diag(w), a)
            q, r = np.linalg.qr(a)
            m_w[i*2:i*2+2, :] = np.linalg.solve(r, q.T)
        # make basis_k2p with fixed k2p
        if k2p is not None:
            basis_k2p[i, :] = input + (k2p - beta[i]) * basis[i, :]

    # savemat('/Users/Himiko/data/niftypad_paper_data/TACs/PPET_30/TACs_30/QM/inputf1.mat',
    #         {'inputf1': inputf1,
    #          'expo': np.exp(-beta[0] * t1),
    #          'convi_ni': np.convolve(inputf1, np.exp(-beta[0]*t1)),
    #          'basis_i': basis[0, :],
    #          'C_r': input,
    #          'coeff': (k2p - beta[0])
    #          }
    #         )


    if w is None:
        basis_w = basis
        m_w = m

    basis = {'beta': beta, 'basis': basis, 'basis_w': basis_w, 'm': m, 'm_w': m_w, 'basis_k2p': basis_k2p,
             'input': input, 'w': w, 'k2p': k2p, 'dt': dt}

    if fig:
        plt.plot(kt.dt2mft(dt), basis['basis'].T)
        plt.show()
    return basis

