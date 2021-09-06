__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def dt2mft(dt):
    mft = np.mean(dt, axis=0)
    return mft


def dt2tdur(dt):
    tdur = dt[1,] - dt[0,]
    return tdur


def mft2tdur(mft):
    print('mind the gap')
    tdur = np.zeros_like(mft)
    tdur[0] = mft[0] * 2
    for i in range(1, len(mft)):
        tdur[i] = (mft[i] - np.sum(tdur)) * 2
    return tdur


def tdur2dt(tdur):
    print('mind the gap')
    st = np.zeros_like(tdur)
    et = np.zeros_like(tdur)
    st[0] = 0
    et[0] = tdur[0]
    for i in range(1, len(tdur)):
        st[i] = et[i - 1]
        et[i] = st[i] + tdur[i]
    dt = [st, et]
    return dt


def mft2dt(mft):
    print('mind the gap')
    tdur = mft2tdur(mft)
    dt = tdur2dt(tdur)
    return dt


def tdur2mft(tdur):
    print('mind the gap')
    dt = tdur2dt(tdur)
    mft = dt2mft(dt)
    return mft


def dt_has_gaps(dt):
    dt_has_gaps = False
    unique, counts = np.unique(dt, return_counts=True)
    gap_ends = unique[counts == 1]
    if len(gap_ends) > 2:
        dt_has_gaps = True
    return dt_has_gaps


def dt_fill_gaps(dt):
    unique, counts = np.unique(dt, return_counts=True)
    gap_ends = unique[counts == 1]
    if len(gap_ends) > 2:
        gap_ends = gap_ends[1:-1]
        gap_starts_index = range(0, len(gap_ends), 2)
        for i in gap_starts_index:
            # print(gap_ends[i:i+2])
            dt_gap = np.unique(np.floor(np.linspace(gap_ends[i], gap_ends[i + 1],
                                                    10))).astype('int16')
            tdur_to_insert = np.diff(dt_gap)
            dt_to_insert = tdur2dt(tdur_to_insert) + gap_ends[i]
            dt = np.insert(dt, np.where(dt[1,] == gap_ends[i])[0] + 1, dt_to_insert, axis=-1)
    return dt


def tac_dt_fill_coffee_break(tac, dt, fig=False):
    # interpolate tac with coffee break using cubic interpolation
    mft = dt2mft(dt)
    tac_inputf1cubic = interpt1cubic(mft, tac, dt)
    dt_no_gaps = dt_fill_gaps(dt)
    tac_no_gaps = int2dt(tac_inputf1cubic, dt_no_gaps)
    if fig:
        plt.plot(mft, tac, '.')
        plt.plot(dt2mft(dt_no_gaps), tac_no_gaps, 'r')
        plt.show()
    return tac_no_gaps, dt_no_gaps


# # # #


def int2dt(f1, dt):
    # in case dt is not in whole seconds
    dt = np.rint(dt).astype(int)
    tdur = dt2tdur(dt)
    f = np.zeros(tdur.size)
    for j in range(0, tdur.size):
        f[j] = np.sum(f1[dt[0, j]:dt[1, j]]) / tdur[j]
    return f


def interpt1(inputt, inputf, dt):
    # interpolate function
    t1 = np.arange(np.amax(dt))
    if inputt[0] > t1[0]:
        inputt = np.append(t1[0], inputt)
        inputf = np.append(0, inputf)
    inputff = interp1d(inputt, inputf, kind='linear', fill_value='extrapolate')
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
