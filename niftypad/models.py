__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt

from niftypad import kt
from niftypad import kp
from niftypad import basis

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # linear models
# # # # # # # # # # # # # # # # # # # # # # # # # # # #


# srtmb_basis - srtm model for img with pre-calculated basis functions

def srtmb_basis(tac, b):
    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1

    for i in range(0, n_beta):
        theta = np.dot(b['m_w'][i*2:i*2+2, :], b['w']*tac)
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta)
        res = (tac - tacf)*b['w']
        ssq[i] = np.sum(res **2)
    i = np.argmin(ssq)
    theta = np.dot(b['m_w'][i*2:i*2+2, :], b['w']*tac)
    a = np.column_stack((b['input'], b['basis'][i, :]))
    tacf = np.dot(a, theta)

    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    return r1, k2, bp, tacf


def srtmb_basis_para2tac(r1, k2, bp, b):
    tacf = []
    theta = kp.srtm_kp2theta(r1, k2, bp)
    i = np.argwhere(abs(b['beta'] - theta[-1]) < 1e-10).squeeze()
    if not i == []:
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta[:-1])
    return tacf


# srtmb - srtm model for tac, basis functions will be calculated

def srtmb(tac, dt, inputf1, w):
    b = basis.make_basis(inputf1, dt, beta_lim=[10e-6, 10e-1], n_beta=128, w=w)
    r1, k2, bp, tacf = srtmb_basis(tac, b)
    return r1, k2, bp, tacf


# srtmb_asl - srtm model for tac with fixed R1, basis functions will be calculated

def srtmb_asl(tac, dt, inputf1, w, r1):
    b = basis.make_basis(inputf1, dt, beta_lim=[10e-6, 10e-1], n_beta=128, w=w)
    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1
    y = tac - r1*b['input']
    for i in range(0,n_beta):
        theta = r1
        theta = np.append(theta, np.dot(y, b['basis'][i, :])/np.dot(b['basis'][i, :], b['basis'][i, :]))  # w won't make a difference here
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta)
        res = (tac - tacf) * b['w'] # w works here
        ssq[i] = np.sum(res ** 2)
    i = np.argmin(ssq)
    theta = r1
    theta = np.append(theta, np.dot(y, b['basis'][i, :]) / np.dot(b['basis'][i, :], b['basis'][i, :]))
    a = np.column_stack((b['input'], b['basis'][i, :]))
    tacf = np.dot(a, theta)

    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    return r1, k2, bp, tacf


# srtmb_k2p_basis - srtm model for img with fixed k2p and pre-calculated basis functions

def srtmb_k2p_basis(tac, b):
    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1
    for i in range(0, n_beta):
        r1 = np.sum(b['w'] * b['basis_k2p'][i] * tac) / np.sum(b['w'] * b['basis_k2p'][i] ** 2)
        ssq[i] = np.sum(b['w'] * (tac - r1 * b['basis_k2p'][i]) ** 2)

    i = np.argmin(ssq)
    r1 = np.sum(b['w'] * b['basis_k2p'][i] * tac) / np.sum(b['w'] * b['basis_k2p'][i] ** 2)
    tacf = r1 * b['basis_k2p'][i]

    theta = r1
    theta = np.append(theta, r1 * (b['k2p'] - b['beta'][i]))
    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    return r1, k2, bp, tacf


# srtmb_k2p - srtm model for tac with fixed k2p, basis functions will be calculated

def srtmb_k2p(tac, dt, inputf1, w, k2p):
    b = basis.make_basis(inputf1, dt, beta_lim=[10e-6, 10e-1], n_beta=128, w=w, k2p=k2p)
    r1, k2, bp, tacf = srtmb_k2p_basis(tac, b)
    return r1, k2, bp, tacf


# logan_ref_k2p - logan reference plot with fixed k2p for tac

def logan_ref_k2p(tac, dt, inputf1, k2p, linear_phase_start, linear_phase_end, fig=True):
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    input_dt = kt.int2dt(inputf1,dt)
    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)
    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    input_cum = np.cumsum((input_dt[:-1] + input_dt[1:]) / 2 * tdur)
    tac = tac[1:]
    input_dt = input_dt[1:]
    yy = tac_cum / tac
    xx = (input_cum + input_dt / k2p) / tac
    tt = np.logical_and(mft > linear_phase_start, mft < linear_phase_end)
    tt = tt[1:]
    dvr, inter, _,  _,  _ = linregress(xx[tt], yy[tt])
    bp = dvr - 1
    yyf = dvr * xx + inter
    if fig:
        plt.plot(xx, yy, 'o')
        plt.plot(xx[tt], yyf[tt], 'r')
        plt.show()
    return bp


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # non-linear models
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# srtm - srtm model for tac, with non-linear optimisation

def srtm_fun(inputf1_dt, r1, k2, bp):
    inputf1, dt = inputf1_dt
    t1 = np.arange(np.amax(dt))
    theta = kp.srtm_kp2theta(r1, k2, bp)
    tac1 = theta[0]*inputf1
    tac1 += theta[1]*np.convolve(inputf1, np.exp(-theta[2]*t1))[0:tac1.size]
    tac = kt.int2dt(tac1, dt)
    return tac


def srtm_para2tac(r1, k2, bp, inputf1_dt):
    tac = srtm_fun(inputf1_dt, r1, k2, bp)
    return tac


def srtm_fun_w(inputf1_dt_w, r1, k2, bp):
    inputf1, dt, w = inputf1_dt_w
    tac = srtm_fun((inputf1, dt), r1, k2, bp)
    if w is None:
        w = 1
    tac_w = tac*w
    return tac_w


def srtm(tac, dt, inputf1, w):
    inputf1_dt = inputf1, dt
    inputf1_dt_w = inputf1, dt, w
    if w is None:
        w = 1
    p, _ = curve_fit(srtm_fun_w, inputf1_dt_w, tac*w, p0=(1, 0.001, 0.5), bounds=(0, [3, 1, 10]))
    r1 = p[0]
    k2 = p[1]
    bp = p[2]
    tacf = srtm_fun(inputf1_dt, r1, k2, bp)
    return r1, k2, bp, tacf


# srtm_k2p - srtm model for tac with fixed k2p, with non-linear optimisation

def srtm_fun_k2p(inputf1_dt_k2p, theta_0, theta_2):
    inputf1, dt, k2p = inputf1_dt_k2p
    inputf1_dt = (inputf1, dt)
    theta_1 = theta_0 * (k2p - theta_2)
    theta = np.array([theta_0, theta_1, theta_2])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    tac = srtm_fun(inputf1_dt, r1, k2, bp)
    return tac


def srtm_fun_k2p_w(inputf1_dt_k2p_w, theta_0, theta_2):
    inputf1, dt, k2p, w = inputf1_dt_k2p_w
    tac = srtm_fun_k2p((inputf1, dt, k2p), theta_0, theta_2)
    if w is None:
        w = 1
    tac_w = tac*w
    return tac_w


def srtm_k2p(tac, dt, inputf1, w, k2p):
    inputf1_dt_k2p = inputf1, dt, k2p
    inputf1_dt_k2p_w = inputf1, dt, k2p, w
    if w is None:
        w = 1
    p, _ = curve_fit(srtm_fun_k2p_w, inputf1_dt_k2p_w, tac*w, p0=(1, 0.5), bounds=(0, [3, 10]))
    theta_0 = p[0]
    theta_2 = p[1]
    theta_1 = theta_0 * (k2p - theta_2)
    theta = np.array([theta_0, theta_1, theta_2])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    tacf = srtm_fun_k2p(inputf1_dt_k2p, theta_0, theta_2)
    return r1, k2, bp, tacf


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #  model for ref input
# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def feng_srtm_fun(ts_te_w, a0, a1, a2, a3, b0, b1, b2, b3):
    ts, te, w = ts_te_w
    if w is None:
        w = 1
    cr_dt_fun = (a0 * a3 * (b0 ** 2 * te / (
    b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(b0 * te)) - b0 * b3 * te / (
                            b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te)) + 2 * b0 / (
                            b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te)) - b3 / (
                            b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te))) - a0 * a3 * (b0 ** 2 * ts / (
    b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(b0 * ts)) - b0 * b3 * ts / (
                                                        b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(b0 * ts)) + 2 * b0 / (
                                                        b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(b0 * ts)) - b3 / (
                                                        b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(
                                                            b0 * ts))) + a0 * a3 * np.exp(-b3 * ts) / (
                 b3 * (b0 ** 2 - 2 * b0 * b3 + b3 ** 2)) - a0 * a3 * np.exp(-b3 * te) / (
                 b3 * (b0 ** 2 - 2 * b0 * b3 + b3 ** 2)) - a1 * a3 * np.exp(-b1 * ts) / (
                 b1 ** 2 - b1 * b3) + a1 * a3 * np.exp(-b1 * te) / (b1 ** 2 - b1 * b3) + a1 * a3 * np.exp(-b0 * ts) / (
                 b0 ** 2 - b0 * b3) - a1 * a3 * np.exp(-b0 * te) / (b0 ** 2 - b0 * b3) + a1 * a3 * np.exp(-b3 * ts) / (
                 b3 * (b1 - b3)) - a1 * a3 * np.exp(-b3 * te) / (b3 * (b1 - b3)) - a1 * a3 * np.exp(-b3 * ts) / (
                 b3 * (b0 - b3)) + a1 * a3 * np.exp(-b3 * te) / (b3 * (b0 - b3)) - a2 * a3 * np.exp(-b2 * ts) / (
                 b2 ** 2 - b2 * b3) + a2 * a3 * np.exp(-b2 * te) / (b2 ** 2 - b2 * b3) + a2 * a3 * np.exp(-b0 * ts) / (
                 b0 ** 2 - b0 * b3) - a2 * a3 * np.exp(-b0 * te) / (b0 ** 2 - b0 * b3) + a2 * a3 * np.exp(-b3 * ts) / (
                 b3 * (b2 - b3)) - a2 * a3 * np.exp(-b3 * te) / (b3 * (b2 - b3)) - a2 * a3 * np.exp(-b3 * ts) / (
                 b3 * (b0 - b3)) + a2 * a3 * np.exp(-b3 * te) / (b3 * (b0 - b3))) / (te - ts)

    return cr_dt_fun*w


def feng_fun_t(t, a0, a1, a2, b0, b1, b2):
    cp = a0*t*np.exp(-b0*t) + a1*np.exp(-b1*t) - a1*np.exp(-b0*t) + a2*np.exp(-b2*t) - a2*np.exp(-b0*t)
    return cp


def feng_srtm_fun_t(t, a0, a1, a2, a3, b0, b1, b2, b3):
    cr = a0 * a3 * (
    -b0 * t * np.exp(b3 * t) / (b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(b0 * t)) + b3 * t * np.exp(
        b3 * t) / (b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(b0 * t)) - np.exp(b3 * t) / (
    b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(b0 * t))) * np.exp(-b3 * t) + a0 * a3 * np.exp(
        -b3 * t) / (b0 ** 2 - 2 * b0 * b3 + b3 ** 2) - a1 * a3 / (b1 * np.exp(b1 * t) - b3 * np.exp(b1 * t)) + a1 * a3 / (
    b0 * np.exp(b0 * t) - b3 * np.exp(b0 * t)) + a1 * a3 * np.exp(-b3 * t) / (b1 - b3) - a1 * a3 * np.exp(-b3 * t) / (
    b0 - b3) - a2 * a3 / (b2 * np.exp(b2 * t) - b3 * np.exp(b2 * t)) + a2 * a3 / (
    b0 * np.exp(b0 * t) - b3 * np.exp(b0 * t)) + a2 * a3 * np.exp(-b3 * t) / (b2 - b3) - a2 * a3 * np.exp(-b3 * t) / (b0 - b3)
    return cr


def feng_srtm(tac, dt, w, fig):
    ts_te_w = (dt[0, ], dt[1, ], w)
    p0 = (3.90671734e+00, 4.34910151e+02, 9.22189828e+01, 1.35949657e-02, 4.56109635e-02, 4.53841116e-02, 4.54180443e-02, 7.71163349e-04)
    p, _ = curve_fit(feng_srtm_fun, ts_te_w, tac*w, p0)
    a0, a1, a2, a3, b0, b1, b2, b3 = p
    t1 = np.arange(np.amax(dt))
    tac1f = feng_srtm_fun_t(t1, a0, a1, a2, a3, b0, b1, b2, b3)
    if fig:
        print(p)
        cp1f = feng_fun_t(t1, a0, a1, a2, b0, b1, b2)
        mft = kt.dt2mft(dt)
        plt.plot(t1, cp1f, 'r', t1, tac1f, 'b', mft, tac, 'go')
        plt.show()
    return tac1f, p