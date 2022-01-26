import numpy as np
from numba import jit, objmode
from scipy.signal import find_peaks

@jit(nopython = True, fastmath = True)
def poincare_max_peaks(traj, idxs = np.empty(1, dtype = np.intc)):
    if idxs.size == 0:
        idxs = np.arange(traj.shape[0])
    poincare_map = []
    peaks_all = []
    for j in idxs:
        with objmode(peaks = 'int64[:]'):
            peaks, tmp = find_peaks(traj[j])
        pmap = np.zeros(peaks.size)
        for i, peak in enumerate(peaks):
            with objmode(polyvals = 'double[:]'):
                polyvals = np.polyfit(np.arange(3), traj[j,peak-1:peak+2], 2)
            if polyvals[0] == 0:
                if polyvals[1] <= 0:
                    pmap[i] = polyvals[2]
                else:
                    pmap[i] = polyvals[2] + polyvals[1]*2
            else:
                pmap[i] = polyvals[2] - polyvals[1]**2.0/(4*polyvals[0])
        poincare_map.append(pmap)
        peaks_all.append(peaks)
    return poincare_map, peaks_all
