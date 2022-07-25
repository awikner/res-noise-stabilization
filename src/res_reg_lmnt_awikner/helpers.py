import numpy as np
from numba import jit, objmode
from scipy.signal import find_peaks
import os


def get_windows_path(path_base):
    path = path_base.replace('/', '\\')
    path = u"\\\\?\\" + path
    return path


@jit(nopython=True, fastmath=True)
def poincare_max(traj, idxs=np.empty(1, dtype=np.intc)):
    if idxs.size == 0:
        idxs = np.arange(traj.shape[0])
    poincare_map = []
    for j in idxs:
        with objmode(peaks='int64[:]'):
            peaks, tmp = find_peaks(traj[j])
        pmap = np.zeros(peaks.size)
        for i, peak in enumerate(peaks):
            with objmode(polyvals='double[:]'):
                polyvals = np.polyfit(np.arange(3), traj[j, peak - 1:peak + 2], 2)
            if polyvals[0] == 0:
                if polyvals[1] <= 0:
                    pmap[i] = polyvals[2]
                else:
                    pmap[i] = polyvals[2] + polyvals[1] * 2
            else:
                pmap[i] = polyvals[2] - polyvals[1] ** 2.0 / (4 * polyvals[0])
        poincare_map.append(pmap)
    return poincare_map


@jit(nopython=True, fastmath=True)
def poincare_max_peaks(traj, idxs=np.empty(1, dtype=np.intc)):
    if idxs.size == 0:
        idxs = np.arange(traj.shape[0])
    poincare_map = []
    peaks_all = []
    for j in idxs:
        with objmode(peaks='int64[:]'):
            peaks, tmp = find_peaks(traj[j])
        pmap = np.zeros(peaks.size)
        for i, peak in enumerate(peaks):
            with objmode(polyvals='double[:]'):
                polyvals = np.polyfit(np.arange(3), traj[j, peak - 1:peak + 2], 2)
            if polyvals[0] == 0:
                if polyvals[1] <= 0:
                    pmap[i] = polyvals[2]
                else:
                    pmap[i] = polyvals[2] + polyvals[1] * 2
            else:
                pmap[i] = polyvals[2] - polyvals[1] ** 2.0 / (4 * polyvals[0])
        poincare_map.append(pmap)
        peaks_all.append(peaks)
    return poincare_map, peaks_all


def set_numba(root_folder, disable_jit):
    config_file = open(os.path.join(root_folder, '.numba_config.yaml'), 'w')
    config_file.write('---\n')
    config_file.write('disable_jit: %d' % int(disable_jit))
    config_file.close()

def get_filename(run_folder_name, var_str, res_itr, train_seed, noise_val, reg_train_time, **kwargs):
    if 'test_idx' in kwargs:
        if 'reg' in kwargs:
            filename = os.path.join(run_folder_name, '%s_res%d_train%d_test%d_noise%e_reg%e_regtrain%d.csv' % (
                var_str, res_itr, train_seed, kwargs['test_idx'], noise_val, kwargs['reg'], reg_train_time))
        else:
            filename = os.path.join(run_folder_name, '%s_res%d_train%d_test%d_noise%e_regtrain%d.csv' % (
                var_str, res_itr, train_seed, kwargs['test_idx'], noise_val, reg_train_time))
    else:
        filename = os.path.join(run_folder_name, '%s_res%d_train%d_noise%e_regtrain%d.csv' % (
            var_str, res_itr, train_seed, noise_val, reg_train_time))
    if 'just_file' in kwargs:
        if isinstance(kwargs['just_file'], bool):
            if kwargs['just_file']:
                return os.path.basename(filename)
            else:
                return filename
        else:
            raise TypeError()
    else:
        return filename