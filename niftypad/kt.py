__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def dt2mft(dt):
    mft = np.mean(dt, axis=0)
    return mft


def dt2tdur(dt):
    tdur = dt[1, ] - dt[0, ]
    return tdur


def mft2tdur(mft):
    # print('mind the gap')
    tdur = np.zeros_like(mft)
    tdur[0] = mft[0] * 2
    for i in range(1, len(mft)):
        tdur[i] = (mft[i] - np.sum(tdur)) * 2
    return tdur


def tdur2dt(tdur):
    # print('mind the gap')
    st = np.zeros_like(tdur)
    et = np.zeros_like(tdur)
    st[0] = 0
    et[0] = tdur[0]
    for i in range(1, len(tdur)):
        st[i] = et[i-1]
        et[i] = st[i] + tdur[i]
    dt = np.array([st, et])
    return dt


def mft2dt(mft):
    # print('mind the gap')
    tdur = mft2tdur(mft)
    dt = tdur2dt(tdur)
    dt = np.rint(dt).astype(int)
    return dt


def tdur2mft(tdur):
    # print('mind the gap')
    dt = tdur2dt(tdur)
    mft = dt2mft(dt)
    return mft


def mft_tdur2dt(mft, tdur):

    tdur_dummy = mft2tdur(mft)
    index = np.where(tdur_dummy-tdur != 0)
    dt = mft2dt(mft)
    for i in index:
        dt[0, i] = mft[i] - tdur[i] / 2
        dt[1, i] = mft[i] + tdur[i] / 2
    dt = np.round(dt)
    dt = np.rint(dt).astype(int)
    return dt

def dt_has_gaps(dt):
    dt_has_gaps = False
    unique, counts = np.unique(dt, return_counts=True)
    gap_ends = unique[counts == 1]
    if len(gap_ends) > 2:
        dt_has_gaps = True
    return dt_has_gaps


def dt_find_gaps(dt):
    dt_gaps = []
    unique, counts = np.unique(dt, return_counts=True)
    gap_ends = unique[counts == 1]
    if len(gap_ends) > 2:
        gap_ends = gap_ends[1:-1]
        gap_starts_index = range(0, len(gap_ends), 2)
        for i in gap_starts_index:
            dt_gaps.append(list(gap_ends[i:i+2]))
    return dt_gaps


def dt_fill_gaps(dt):
    unique, counts = np.unique(dt, return_counts=True)
    gap_ends = unique[counts == 1]
    if len(gap_ends) > 2:
        gap_ends = gap_ends[1:-1]
        gap_starts_index = range(0, len(gap_ends), 2)
        for i in gap_starts_index:
            # print(gap_ends[i:i+2])
            dt_gap = np.unique(np.floor(np.linspace(gap_ends[i], gap_ends[i+1], 10+2))).astype('int16')
            tdur_to_insert = np.diff(dt_gap)
            dt_to_insert = tdur2dt(tdur_to_insert) + gap_ends[i]
            dt = np.insert(dt, np.where(dt[1, ] == gap_ends[i])[0]+1, dt_to_insert, axis=-1)
    return dt


def tac_dt_fill_coffee_break(tac, dt, interp, fig=False):
    # interpolate tac with coffee break using linear or cubic interpolation
    mft = dt2mft(dt)
    dt_no_gaps = dt_fill_gaps(dt)
    mft_no_gaps = dt2mft(dt_no_gaps)
    if interp is 'linear':
        tac_inputf1linear = interpt1(mft, tac, dt)
        tac_no_gaps = int2dt(tac_inputf1linear, dt_no_gaps)
    if interp is 'cubic':
        tac_inputf1cubic = interpt1cubic(mft, tac, dt)
        tac_no_gaps = int2dt(tac_inputf1cubic, dt_no_gaps)
    if interp is 'linear':
        tac_inputf1linear = interpt1(mft, tac, dt)
        tac_no_gaps = int2dt(tac_inputf1linear, dt_no_gaps)
    if interp is 'zero':
        tac_no_gaps = np.zeros_like(mft_no_gaps)
    if fig:
        plt.plot(mft, tac, '.')
        plt.plot(dt2mft(dt_no_gaps), tac_no_gaps, 'r')
        plt.show()
    # replace tac with the original values without interpolation
    _, index_no_gaps, index = np.intersect1d(mft_no_gaps, mft, return_indices=True)
    tac_no_gaps[index_no_gaps] = tac[index]
    return tac_no_gaps, dt_no_gaps
# # # #


def int2dt(f1, dt):
    # in case dt is not in whole seconds
    dt = np.rint(dt).astype(int)
    tdur = dt2tdur(dt)
    f = np.zeros(tdur.size)
    for j in range(0, tdur.size):
        f[j] = np.sum(f1[dt[0, j]:dt[1, j]])/tdur[j]
    return f


def interpt1(inputt, inputf, dt):
    # interpolate function
    t1 = np.arange(np.amax(dt))
    # if inputt[0] > t1[0]:
    #     inputt = np.append(t1[0], inputt)
    #     inputf = np.append(0, inputf)
    inputt = np.append(0, inputt)
    inputf = np.append(0, inputf)
    inputff = interp1d(inputt, inputf, kind='linear', fill_value='extrapolate')

    # # Qmodelling interpolation
    # inputff = interp1d(inputt, inputf, kind='linear', fill_value=(inputf[0], inputf[-1]),bounds_error=False)

    inputf1 = inputff(t1)
    return inputf1


def interpt1cubic(inputt, inputf, dt):
    # interpolate function
    t1 = np.arange(np.amax(dt))
    if inputt[0] > t1[0]:
        inputt = np.append(t1[0], inputt)
        inputf = np.append(0, inputf)
    inputff = interp1d(inputt, inputf, kind='cubic', fill_value='extrapolate')
    inputf1 = inputff(t1)
    return inputf1