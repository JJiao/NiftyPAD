__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np
from scipy.interpolate import interp1d


def dt2mft(dt):
    mft = np.mean(dt, axis=0)
    return mft


def dt2tdur(dt):
    tdur = dt[1, ] - dt[0, ]
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
        st[i] = et[i-1]
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
    inputf1 = np.interp(x=t1, xp=inputt, fp=inputf, left=0)
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