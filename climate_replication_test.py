#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
# Assume will be finished in no more than 18 hours
# SBATCH -t 4:00:00
# Launch on 12 cores distributed over as many nodes as needed
# SBATCH --ntasks=1
# Assume need 6 GB/core (6144 MB/core)
# SBATCH --mem-per-cpu=6144
# SBATCH --mail-user=awikner1@umd.edu
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

from numba.core.errors import NumbaPerformanceWarning
import warnings
from itertools import product
import ray
import sys
import getopt
import os

from datetime import datetime
# from lorenzrungekutta_numba import fx
# from lorenzrungekutta_numba import fy
# from lorenzrungekutta_numba import fz
import numpy as np
# from sklearn.linear_model import Ridge
from scipy.linalg import solve, solve_sylvester
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from numba import jit, njit, objmode
from numba.experimental import jitclass
from numba.types import int32, int64, double
import time
import cProfile
import pstats

import pkg_resources
import os
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(
    ["%s==%s" % (i.key, i.version) for i in installed_packages])
isray = [('ray==' in elem) for elem in installed_packages_list]
if (True in isray):
    print('Ray installed')
else:
    os.system('pip install -r -U ray')

sys.path.append('/h/awikner/res-noise-stabilization/')
from lorenzrungekutta_numba import *
from ks_etdrk4 import *
from csc_mult import *
from poincare_max import *

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

@njit
def str_to_int(s):
    # Converts a string to an int in numba compiled functions
    final_index, result = len(s) - 1, 0
    for i, v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result


@jit(nopython=True, fastmath=True)
def mean_numba_axis1(mat):
    # Computes the mean over axis 1 in numba compiled functions
    res = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        res[i] = np.mean(mat[i])

    return res


@jit(nopython=True, fastmath=True)
def sum_numba_axis0(mat):
    # Computes the sum over axis 0 in numba compiled functions
    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.sum(mat[:, i])
    return res


@jit(nopython=True, fastmath=True)
def wasserstein_distance_empirical(measured_samples, true_samples):
    # Computes the wasserstein distance between the empirical CDFs of the two input sets of samples. Faster than scipy.
    if np.any(np.isnan(measured_samples)):
        return np.NAN
    if np.any(np.isinf(measured_samples)):
        return np.inf
    measured_samples.sort()
    true_samples.sort()
    n, m, n_inv, m_inv = (measured_samples.size, true_samples.size,
                          1/measured_samples.size, 1/true_samples.size)
    n_itr = 0; m_itr = 0; measured_cdf = 0; true_cdf = 0; wass_dist = 0
    if measured_samples[n_itr] < true_samples[m_itr]:
        prev_sample = measured_samples[n_itr]
        measured_cdf += n_inv
        n_itr += 1
    elif true_samples[m_itr] < measured_samples[n_itr]:
        prev_sample = true_samples[m_itr]
        true_cdf += m_inv
        m_itr += 1
    else:
        prev_sample = true_samples[m_itr]
        measured_cdf += n_inv; true_cdf += m_inv
        n_itr += 1; m_itr += 1
    while n_itr < n and m_itr < m:
        if measured_samples[n_itr] < true_samples[m_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (measured_samples[n_itr]-prev_sample))
            prev_sample = measured_samples[n_itr]
            measured_cdf += n_inv
            n_itr += 1
        elif true_samples[m_itr] < measured_samples[n_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (true_samples[m_itr]-prev_sample))
            prev_sample = true_samples[m_itr]
            true_cdf += m_inv
            m_itr += 1
        else:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (true_samples[m_itr]-prev_sample))
            prev_sample = true_samples[m_itr]
            measured_cdf += n_inv; true_cdf += m_inv
            n_itr += 1; m_itr += 1
    if n_itr == n:
        for itr in range(m_itr, m):
            wass_dist += np.abs((1.0 - true_cdf) *
                                (true_samples[itr] - prev_sample))
            prev_sample = true_samples[itr]
            true_cdf += m_inv
    else:
        for itr in range(n_itr, n):
            wass_dist += np.abs((measured_cdf - 1.0) *
                                (measured_samples[itr] - prev_sample))
            prev_sample = measured_samples[itr]
            measured_cdf += n_inv

    return wass_dist


class Reservoir:
    def __init__(self, rk, res_gen, input_size, rsvr_size=300, spectral_radius=0.6, input_weight=1, leakage=1.0, win_type='full', bias_type='old', avg_degree = 10):
        # Define class for storing reservoir layers generated from input parameters and an input random number generator
        self.rsvr_size = rsvr_size
        """
        print('Spectral Radius: %0.2f' % spectral_radius)
        print('Input Weight: %0.2f' % input_weight)
        print('Leakage: %0.3f' % leakage)
        print('Win Type: %s' % win_type)
        print('Bias type: %s' % bias_type)
        """

        density = avg_degree/rsvr_size
        unnormalized_W = (res_gen.random((rsvr_size, rsvr_size))*2 - 1)
        for i in range(unnormalized_W[:, 0].size):
            for j in range(unnormalized_W[0].size):
                if res_gen.random(1) > 10/rsvr_size:
                    unnormalized_W[i][j] = 0

        max_eig = eigs(unnormalized_W, k=1,
                       return_eigenvectors=False, maxiter=10**5)

        W_sp = csc_matrix(np.ascontiguousarray(
            spectral_radius/np.abs(max_eig[0])*unnormalized_W))
        self.W_data, self.W_indices, self.W_indptr, self.W_shape = \
                (W_sp.data, W_sp.indices, W_sp.indptr, np.array(list(W_sp.shape)))
        """
        print('Adjacency matrix section:')
        print(self.W_data[:4])
        """

        if win_type == 'full':
            input_vars = np.arange(input_size)
        elif win_type == 'x':
            input_vars = np.array([0])
        if bias_type == 'old':
            const_frac = 0.15
            const_conn = int(rsvr_size*const_frac)
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:const_conn, 0] = (res_gen.random(
                Win[:const_conn, 0].size)*2 - 1)*input_weight
            q = int((rsvr_size-const_conn)//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[const_conn+q*i:const_conn+q *
                    (i+1), var+1] = (res_gen.random(q)*2-1)*input_weight
        elif bias_type == 'new_random':
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:, 0] = (res_gen.random(rsvr_size)*2-1)*input_weight
            q = int(rsvr_size//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[q*i:q*(i+1), var+1] = (res_gen.random(q)*2-1)*input_weight
            leftover_nodes = rsvr_size - q*input_vars.size
            var = input_vars[res_gen.integers(
                input_vars.size, size=leftover_nodes)]
            Win[rsvr_size-leftover_nodes:, var +
                1] = (res_gen.random(leftover_nodes)*2-1)*input_weight
        elif bias_type == 'new_new_random':
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:, 0] = (res_gen.random(rsvr_size)*2-1)*input_weight
            q = int(rsvr_size//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[q*i:q*(i+1), var+1] = (res_gen.random(q)*2-1)*input_weight
            leftover_nodes = rsvr_size - q*input_vars.size
            var = input_vars[res_gen.choice(
                input_vars.size, size=leftover_nodes, replace=False)]
            Win[rsvr_size-leftover_nodes:, var +
                1] = (res_gen.random(leftover_nodes)*2-1)*input_weight
        elif bias_type == 'new_const':
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:, 0] = input_weight
            q = int(rsvr_size//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[q*i:q*(i+1), var+1] = (res_gen.random(q)*2-1)*input_weight
            leftover_nodes = rsvr_size - q*input_vars.size
            var = input_vars[res_gen.integers(
                input_vars.size, size=leftover_nodes)]
            Win[rsvr_size-leftover_nodes:, var +
                1] = (res_gen.random(leftover_nodes)*2-1)*input_weight

        self.Win = np.ascontiguousarray(Win)

        self.X = (res_gen.random((rsvr_size, rk.train_length+2))*2 - 1)
        self.Wout = np.array([])
        self.leakage = leakage
        """
        print('Win Section:')
        print(Win[:3,:3])
        """

class RungeKutta:
    def __init__(self, x0=2, y0=2, z0=23, h=0.01, tau=0.1, T=300, ttsplit=5000, u0=0, system='lorenz', params=np.array([[], []], dtype=np.complex128)):
        # Class for obtaining training and testing dynamical system time series data
        if system == 'lorenz':
            int_step = int(tau/h)
            u_arr = np.ascontiguousarray(rungekutta(
                x0, y0, z0, h, T, tau)[:, ::int_step])
            self.input_size = 3

            u_arr[0] = (u_arr[0] - 0)/7.929788629895004
            u_arr[1] = (u_arr[1] - 0)/8.9932616136662
            u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
            self.params = params

        elif system == 'KS':
            u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, params=params)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
        else:
            raise ValueError

        self.train_length = ttsplit
        self.u_arr_train = u_arr[:, :ttsplit+1]

        # u[ttsplit], the (ttsplit + 1)st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]


@jit(nopython=True, fastmath=True)
def RungeKuttawrapped(x0=2, y0=2, z0=23, h=0.01, tau=0.1, T=300, ttsplit=5000, u0=0, system='lorenz', params=np.array([[], []], dtype=np.complex128)):
    # Numba function for obtaining training and testing dynamical system time series data
    if system == 'lorenz':
        int_step = int(tau/h)
        u_arr = np.ascontiguousarray(rungekutta(
            x0, y0, z0, h, T, tau)[:, ::int_step])

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict(u0, tau=tau, T=T, params=params)
        u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    # size 5001

    # noisy training array
    # switch to gaussian

    # plt.plot(self.u_arr_train_noise[0, :500])

    # u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params


@jit(nopython=True, fastmath=True)
def RungeKuttawrapped_pred(h=0.01, tau=0.1, T=300, ttsplit=5000, u0_array=np.array([[], []], dtype=np.complex128), system='lorenz', params=np.array([[], []], dtype=np.complex128)):
    if system == 'lorenz':
        int_step = int(tau/h)
        u_arr = np.ascontiguousarray(
            rungekutta_pred(u0_array, h, tau, int_step))
        # self.train_length = ttsplit
        # self.noise_scaling = noise_scaling

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict_pred(
            u0_array, tau=tau, T=T, params=params)
        u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    # size 5001

    # noisy training array
    # switch to gaussian

    # plt.plot(self.u_arr_train_noise[0, :500])

    # u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params


def getX(res, rk, x0=1, y0=1, z0=1):
    u_training = rk.u_arr_train
    res.X = getXwrapped(np.ascontiguousarray(u_training), res.X, res.Win,
                        res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage)

    return res.X
# takes a reservoir object res along with initial conditions


@jit(nopython=True, fastmath=True)
def getXwrapped(u_training, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise, noisetype='none', noise_scaling=0, noise_realization=0, traintype='normal'):

    # loops through every timestep
    if noisetype in ['gaussian', 'perturbation']:
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq'] or 'confined' in traintype:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i]+noise[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]+noise[:, i]))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i])+noise[:, i])
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif noisetype not in ['gaussian_onestep', 'perturbation_onestep'] and \
            ('gaussian' in noisetype and 'step' in noisetype):
        noise_steps = str_to_int(noisetype.replace('gaussian', '').replace(
            'perturbation', '').replace('step', ''))
        res_X_nonoise = np.copy(res_X)
        for i in range(0, u_training[0].size):
            res_X_nonoise[:, i+1] = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(Win @ np.append(
                1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq']:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i]+noise[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                        1., u_training[:, i+k]+noise[:, i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                res_X[:, i+noise_steps] = temp_x
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i]))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                        1., u_training[:, i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i+k]))
                res_X[:, i+noise_steps] = temp_x
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i])
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                        1., u_training[:, i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i+k])
                res_X[:, i+noise_steps] = temp_x
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif noisetype in ['gaussian_onestep', 'perturbation_onestep']:
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq']:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i]+noise[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                temp_x = np.tanh(Win @ np.append(1., u_training[:, i])+mult_vec(
                    W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i]))
                temp_x = np.tanh(Win @ np.append(1., u_training[:, i])+mult_vec(
                    W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                    1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i])
                temp_x = np.tanh(Win @ np.append(1., u_training[:, i])+mult_vec(
                    W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif 'gaussian' in noisetype:
        noise_steps = str_to_int(noisetype.replace('gaussian', ''))

        # noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_gen, noise_realization)
        res_X_nonoise = np.copy(res_X)
        for i in range(0, u_training[0].size):
            res_X_nonoise[:, i+1] = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(Win @ np.append(
                1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        for i in range(0, u_training[0].size-noise_steps):
            temp_x = res_X_nonoise[:, i]
            for k in range(noise_steps):
                if k == 0:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                        1., u_training[:, i+k]+noise[:, i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                else:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(
                        1., u_training[:, i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            res_X[:, i+noise_steps] = temp_x
        u_training_wnoise = u_training+noise
        return res_X, u_training_wnoise

    elif traintype in ['sylvester_wD'] or 'gradient' in traintype:
        res_D = np.zeros((res_X.shape[0], u_training.shape[1]+1))
        for i in range(0, u_training[0].size):
            res_internal = Win @ np.append(1., u_training[:, i])+mult_vec(
                W_data, W_indices, W_indptr, W_shape, res_X[:, i])
            res_X[:, i+1] = (1.0 - leakage)*res_X[:, i] + \
                             leakage*np.tanh(res_internal)
            res_D[:, i+1] = leakage/(np.power(np.cosh(res_internal), 2.0))
        return res_X, res_D
    else:
        for i in range(0, u_training[0].size):
            res_X[:, i+1] = (1.0 - leakage)*res_X[:, i] + leakage*np.tanh(Win @ np.append(
                1., u_training[:, i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]))
        return res_X, u_training


"""
@jit(nopython = True, fastmath = True)
def getjacobian(Win, W, Wout, Dn):
    jacsize  = Wout.shape[1]-1
    res_size = Win.shape[0]
    input_size = Win.shape[1]-1
    jacobian = np.zeros((jacsize, jacsize))
    bottom_right = np.zeros((res_size + input_size, input_size))
    bottom_right[res_size:, :] = np.identity(input_size)
    for i in range(Dn.shape[1]):
        D    = np.diag(Dn[:,i])
        DW   = numba_matT_CSR_mult(D,W)
        DWin = D @ Win[:,1:]
        bottom_right[:res_size,:input_size] = DWin
        jacobian[:res_size, :res_size] += DW
        jacobian[res_size:, :res_size] += Wout[:,1:res_size+1] @ DW
        jacobian[:res_size, res_size:] += DWin
        jacobian[res_size:, res_size:] += Wout[:,1:] @ bottom_right
    return jacobian
"""


def gen_noise_driver(data_shape, res_shape, traintype, noisetype, noise_scaling, noise_stream, noise_realizations):
    if noisetype in ['gaussian', 'perturbation']:
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq'] or 'confined' in traintype:
            noise = gen_noise(data_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        else:
            raise ValueError
    elif noisetype not in ['gaussian_onestep', 'perturbation_onestep'] and \
            ('gaussian' in noisetype and 'step' in noisetype):
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq']:
            noise = gen_noise(data_shape[0], data_shape[1], str(
                noisetype), noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        else:
            raise ValueError
    elif noisetype in ['gaussian_onestep', 'perturbation_onestep']:
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq']:
            noise = gen_noise(data_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], noisetype,
                              noise_scaling, noise_stream, noise_realizations)
        else:
            raise ValueError
    elif 'gaussian' in noisetype:
        noise = gen_noise(data_shape[0], data_shape[1], str(
            noisetype), noise_scaling, noise_stream, noise_realizations)
    elif 'gradient' in noisetype or noisetype=='none':
        noise = np.zeros((1, data_shape[0], data_shape[1]))
    else:
        raise ValueError
    return noise


# @jit(nopython = True, fastmath = True)
def gen_noise(noise_size, noise_length, noisetype, noise_scaling, noise_stream, noise_realizations):
    noise = np.zeros((noise_realizations, noise_size, noise_length))
    if 'gaussian' in noisetype:
        for i in range(noise_realizations):
            noise[i] = noise_stream[i].standard_normal(
                (noise_size, noise_length))*noise_scaling
    if noisetype in ['perturbation', 'perturbation_onestep']:
        for noise_realization in noise_realizations:
            if noise_realization < noise_size:
                noise[noise_realization, noise_realization] = np.ones(
                    noise_length)*noise_scaling
            elif noise_realization < 2*noise_length:
                noise[noise_realization, noise_realization -
                    noise_size] = -np.ones(noise_length)*noise_scaling
            else:
                raise ValueError

    return noise


def get_states(res, rk, noise, noisetype='none', noise_scaling=0, noise_realizations=1,
        traintype='normal', skip=150):
    if traintype == 'getD':
        Dn = getD(np.ascontiguousarray(rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype)
        return Dn
    elif traintype not in ['sylvester', 'sylvester_wD']:
        res.data_trstates, res.states_trstates, res.gradient_reg = get_states_wrapped(
            np.ascontiguousarray(
                rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype)
    else:
        res.data_trstates, res.states_trstates, res.left_mat = get_states_wrapped(
            np.ascontiguousarray(
                rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype)


@jit(nopython=True, fastmath=True)
def get_states_wrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, skip, noise, noisetype='none',
        noise_scaling=0, noise_realizations=1, traintype='normal', q=0):
    res_X = np.ascontiguousarray(res_X)
    u_arr_train = np.ascontiguousarray(u_arr_train)
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
    Y_train = np.ascontiguousarray(u_arr_train[:, skip+1:])
    if traintype in ['normal', 'normalres1', 'normalres2']:
        # Normal multi-noise training that sums all reservoir state outer products
        data_trstates = np.zeros((n, rsvr_size+1+n), dtype=np.float64)
        states_trstates = np.zeros(
            (n+rsvr_size+1, n+rsvr_size+1), dtype=np.float64)
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = X[:, skip+1:(res_d - 1)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            data_trstates += Y_train @ X_train.T
            states_trstates += X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rmeanq','rmeanqres1','rmeanqres2']:
        # Training using the mean and the rescaled sum of the perturbations from the mean
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape,leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
                X_all = np.zeros((X_train.shape[0], X_train.shape[1], noise_realizations))
            X_train_mean += X_train/noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_mean.T
        states_trstates = X_train_mean @ X_train_mean.T
        for i in range(noise_realizations):
            Q_fit = X_all[:, :, i] - X_train_mean
            states_trstates += (Q_fit @ Q_fit.T)/noise_realizations
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rmean', 'rmeanres1', 'rmeanres2']:
        # Training using the mean only
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = X[:, skip+1:(res_d - 1)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
        data_trstates = Y_train @ X_train_mean.T
        states_trstates = X_train_mean @ X_train_mean.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rqmean','rqmeanres1','rqmeanres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the mean
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip:-1]), axis=0))
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape,leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates = X_train_0 @ X_train_0.T
        for i in range(noise_realizations):
            Q_fit = X_all[:,:,i] - X_train_mean
            states_trstates += (Q_fit @ Q_fit.T)/noise_realizations
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rq','rqres1','rqres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the noiseless reservoir
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip:-1]), axis=0))
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape,     leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates = X_train_0 @ X_train_0.T
        for i in range(noise_realizations):
            Q_fit = X_all[:,:,i] - X_train_0
            states_trstates += (Q_fit @ Q_fit.T)/noise_realizations
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rplusq', 'rplusqres1', 'rplusqres2']:
        # Training using the sum of the outer products of the sum of the noiseless reservoir and
        # the perturbation from the mean
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip:-1]), axis=0))
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
            X_all[:, :, i] = X_train
        Y_fit = Y_train
        X_fit = X_all[:, :, 0] - X_train_mean + X_train_0
        for i in range(1, noise_realizations):
            Y_fit = np.append(Y_fit, Y_train, axis=1)
            X_fit = np.append(
                X_fit, X_all[:, :, i] - X_train_mean + X_train_0, axis=1)
        data_trstates = Y_fit @ X_fit.T
        states_trstates = X_fit @ X_fit.T
        return [data_trstates, states_trstates, gradient_reg]
    elif 'confined' in traintype:
        if 'upper' in traintype:
            with objmode(ub='double'):
                ub = float(traintype.replace('confinedupper', ''))
            lb = 0
        elif 'lower' in traintype:
            with objmode(lb='double'):
                lb = float(traintype.replace('confinedlower', ''))
            ub = 0
        else:
            with objmode(ub='double'):
                ub = float(traintype.replace('confined', ''))
            lb = -ub
        data_trstates = np.zeros((n, rsvr_size+1+n), dtype=np.float64)
        states_trstates = np.zeros(
            (n+rsvr_size+1, n+rsvr_size+1), dtype=np.float64)
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, 'normal')
            X = X[:, skip+1:(res_d - 1)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis=0))
            sync_len = 0
            inbound_data = np.where(np.logical_and(
                Y_train[0] < ub, Y_train[0] > lb))[0]

            crossing_points = np.where(
                inbound_data[1:] - inbound_data[:-1] != 1)[0]+1
            if 0 in inbound_data:
                unsynced_inbound_data = inbound_data[sync_len:crossing_points[0]]
                for cp, cp_next in zip(crossing_points[:-1], crossing_points[1:]):
                    unsynced_inbound_data = np.append(
                        unsynced_inbound_data, inbound_data[cp+sync_len:cp_next])
            else:
                unsynced_inbound_data = inbound_data[crossing_points[0] +
                    sync_len:crossing_points[1]]
                for cp, cp_next in zip(crossing_points[1:-1], crossing_points[2:]):
                    unsynced_inbound_data = np.append(
                        unsynced_inbound_data, inbound_data[cp+sync_len:cp_next])
            unsynced_inbound_data = np.append(
                unsynced_inbound_data, inbound_data[crossing_points[-1]+sync_len:])
            Y_train = Y_train[:, unsynced_inbound_data]
            X_train = X_train[:, unsynced_inbound_data]
            """
            print('Bounded training data shape:')
            print(Y_train.shape)
            print(np.max(Y_train[0]))
            print(np.min(Y_train[0]))
            """
        data_trstates += Y_train @ X_train.T
        states_trstates += X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        for i in range(X.shape[1]):
            D_n = np.concatenate(
                (matrix_diag_mult(D[:, i], Win[:, 1:]), np.identity(n)), axis=0)
            gradient_reg[1:,1:] += D_n @ D_n.T
        print('Gradient reg:')
        print(gradient_reg[:5,:5])
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient12':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        """
        print('X')
        print(X[:3,:3])
        print('D')
        print(D[:3,:3])
        print('Train:')
        print(u_arr_train[:3,skip:skip+3])
        print('Win')
        print(Win[:5,:5])
        """
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(
            Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                                np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis=1)
        for i in range(1, X.shape[1]):
            if i > 0:
                D_n2 = np.concatenate((np.zeros((1, n)), matrix_diag_mult(
                    D[:, i-1], Win[:, 1:]), np.identity(n)), axis=0)
                E_n = np.concatenate((np.zeros((1, rsvr_size+n+1)), matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, np.diag(D[:, i])) + leakage_mat,
                    np.zeros((n, rsvr_size+n+1))), axis=0)
                E_nD_n2 = E_n @ D_n2
                """
                if i == 1:
                    print('E_n * D_n')
                    print(E_nD_n2[:5,:5])
                if i == 2:
                    print('Reg components @ 2,0')
                    print(E_nD_n2[:5,:5])
                """
                gradient_reg += E_nD_n2 @ E_nD_n2.T
            D_n = np.concatenate(
                (matrix_diag_mult(D[:, i], Win[:, 1:]), np.identity(n)), axis=0)
            """
            if i == 1:
                print('D_n:')
                print(D_n[:5,:5])
            if i == 2:
                print('Reg components @ 2,1')
                print(D_n[:5,:5])
            """
            gradient_reg[1:, 1:] += D_n @ D_n.T
            """
            if i == 1:
                print('Init Gradient reg:')
                print(gradient_reg[:5,:5])
            if i == 2:
                print('Second Gradient reg:')
                print(gradient_reg[:5,:5])
            """
        """
        print('Gradient reg:')
        print(gradient_reg[:5,:5])
        """
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient2':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(
            Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        for i in range(1, X.shape[1]):
            D_n2 = np.concatenate((np.zeros((1, n)), matrix_diag_mult(
                D[:, i-1], Win[:, 1:]), np.identity(n)), axis=0)
            E_n = np.concatenate((np.zeros((1, rsvr_size+n+1)), matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, np.diag(D[:, i])) +
                np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                               np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis=1),
                np.zeros((n, rsvr_size+n+1))), axis=0)
            E_nD_n2 = E_n @ D_n2
            gradient_reg += E_nD_n2 @ E_nD_n2.T
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif 'gradientk' in traintype:
        k = str_to_int(traintype.replace('gradientk', ''))
        #print(k)
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        """
        print('X')
        print(X[:3,:3])
        print('D')
        print(D[:3,:3])
        print('Train:')
        print(u_arr_train[:3,skip:skip+3])
        print('Win')
        print(Win[:5,:5])
        """
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        D_n = np.zeros((rsvr_size+n+1, n, k))
        E_n = np.zeros((rsvr_size+n+1, rsvr_size+n+1, k-1))
        reg_components = np.zeros((rsvr_size+n+1, n, k))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(
            Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        leakage_mat = np.concatenate((np.zeros(
            (rsvr_size, 1)), (1-leakage)*np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis=1)
        """
        with objmode(tic = 'double'):
            tic = time.perf_counter()
        """
        for i in range(k):
            D_n[1:rsvr_size+1, :, i] = matrix_diag_mult(D[:, i], Win[:, 1:])
            D_n[rsvr_size+1:, :, i] = np.identity(n)
        for i in range(1, k):
            E_n[1:rsvr_size+1, :, i-1] = matrix_diag_sparse_mult_add(
                D[:, i], W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat)
        reg_components[:, :, k-1] = D_n[:, :, -1]
        """
        print('D_n:')
        print(reg_components[:5,:5,k-1])
        """

        for i in range(k-1):
            reg_components[:, :, i] = D_n[:, :, i]
            for j in range(i, k-1):
                reg_components[:, :, i] = matrix_sparse_mult(
                    E_n[:, :, j], reg_components[:, :, i])
        """
        print('E_n * D_n')
        print(reg_components[:5,:5,0])
        """

        for i in range(k, X.shape[1]):
            for j in range(k):
                gradient_reg += reg_components[:,
                    :, j] @ reg_components[:, :, j].T
            """
            if i == k:
                print('Init Gradient reg:')
                print(gradient_reg[:5,:5])
            if i == k+1:
                print('Second Gradient reg:')
                print(gradient_reg[:5,:5])
            """
            E_n[1:rsvr_size+1, :, k-2] = matrix_diag_sparse_mult_add(
                D[:, i], W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat)
            for j in range(k-1):
                reg_components[:, :, j] = matrix_sparse_mult(
                    E_n[:, :, k-2], reg_components[:, :, j+1])
            reg_components[1:rsvr_size+1, :, k - 1] = matrix_diag_mult(D[:, i], Win[:, 1:])
            reg_components[1+rsvr_size:, :, k-1] = np.identity(n)
            """
            if i == k:
                print('Reg components @ 2,1')
                print(reg_components[:5,:5, k -1])
            if i == k:
                print('Reg components @ 2,0')
                print(reg_components[:5,:5, 0])
            """
            """
            if i % 50 == 0:
                with objmode(toc = 'double'):
                    toc = time.perf_counter()
                runtime = toc - tic
                tic = toc
                print('50 Iter Runtime: ')
                print(runtime)
            """
        for j in range(k):
            gradient_reg += reg_components[:, :, j] @ reg_components[:, :, j].T
        """
        print('Gradient reg:')
        print(gradient_reg[:5,:5])
        """
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'sylvester':
        X, p = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[i], noisetype, noise_scaling, 1, traintype)
        X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
        X_train = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0))
        data_trstates = Y_train @ X_train.T
        data_trstates[:, 1:rsvr_size+1] += noise_scaling**2/noise_realizations * \
            matrix_dot_left_T(W_mat_data, W_mat_indices,
                              W_mat_indptr, W_mat_shape, Win[:, 1:])
        states_trstates = X_train @ X_train.T
        left_mat = -noise_scaling**2/noise_realizations * \
            (Win[:, 1:].T @ Win[:, 1:])
        return [data_trstates, states_trstates, left_mat]
    elif traintype == 'sylvester_wD':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[i], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis=0)
        Dmean = mean_numba_axis1(D)
        temp_mat = np.diag(Dmean) @ Win[:, 1:]
        target_correction = matrix_dot_left_T(
            W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, temp_mat)
        left_mat = temp_mat.T @ temp_mat
        target_correction = noise_scaling**2/noise_realizations * target_correction
        left_mat = -noise_scaling**2/noise_realizations * left_mat
        data_trstates = Y_train @ X_train.T
        data_trstates[:, 1:rsvr_size+1] += target_correction
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, left_mat]
    else:
        data_trstates = np.zeros((n, rsvr_size+1+n), dtype=np.float64)
        states_trstates = np.zeros(
            (n+rsvr_size+1, n+rsvr_size+1), dtype=np.float64)
        return [data_trstates, states_trstates, gradient_reg]


@jit(nopython=True, fastmath=True)
def getD(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise, skip, noisetype='none',
         noise_scaling=0, noise_realizations=1, traintype='normal'):
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    Y_train = u_arr_train[:, skip+1:]
    X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices,
                       W_indptr, W_shape, leakage, noise, 'none', 0, 0, 'gradient')
    return [D[:, skip+1:(res_d - 1)]]

# CONCATENATE ALL THE DATA BEFORE RUNNING REGRESSION


def predict(res, u0,  steps=1000):
    Y = predictwrapped(res.X, res.Win, res.W_data, res.W_indices,
                       res.W_indptr, res.W_shape, res.Wout, res.leakage, u0, steps)
    return Y


@jit(nopython=True, fastmath=True)
def predictwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape, Wout, leakage, u0, steps):
    Y = np.empty((Win.shape[1]-1, steps + 1))
    X = np.empty((res_X.shape[0], steps + 1))

    Y[:, 0] = u0
    X[:, 0] = res_X[:, -2]

    for i in range(0, steps):
        # y_in = Y[:,i].reshape(3,1)
        # x_prev = X[:,i].reshape(res.rsvr_size,1)
        X[:, i+1] = (1-leakage)*X[:, i] + leakage*np.tanh(Win @ np.append(1.,
                     Y[:, i]) + mult_vec(W_data, W_indices, W_indptr, W_shape, X[:, i]))
        # X = np.concatenate((X, x_current), axis = 1)
        Y[:, i+1] = Wout @ np.concatenate((np.array([1.]), X[:, i+1], Y[:, i]))
        # y_out = np.matmul(res.Wout, x_current)
        # Y[:,i+1] = y_out

    return Y

# @jit(nopython = True, fastmath = True)


def get_test_data(test_stream, tau, num_tests, rkTime, split, system='lorenz'):
    # np.random.seed(0)
    if system == 'lorenz':
        ic = test_stream[0].random(3)*2-1
        u0 = np.zeros(64)
    elif system == 'KS':
        ic = np.zeros(3)
        u0 = (test_stream[0].random(64)*2-1)*0.6
    u_arr_train_nonoise, u_arr_test, p, params = RungeKuttawrapped(x0=ic[0],
         y0=ic[1], z0=30*ic[2], tau=tau, T=rkTime, ttsplit=split, u0=u0, system=system)
    rktest_u_arr_train_nonoise = np.zeros(
        (u_arr_train_nonoise.shape[0], u_arr_train_nonoise.shape[1], num_tests))
    rktest_u_arr_test = np.zeros(
        (u_arr_test.shape[0], u_arr_test.shape[1], num_tests))
    rktest_u_arr_train_nonoise[:, :, 0] = u_arr_train_nonoise
    rktest_u_arr_test[:, :, 0] = u_arr_test
    """
    print('Test data %d' % 0)
    print(rktest_u_arr_test[-3:,-3:,0])
    """
    for i in range(1, num_tests):
        # np.random.seed(i)
        if system == 'lorenz':
            ic = test_stream[i].random(3)*2-1
            u0 = np.zeros(64)
        elif system == 'KS':
            ic = np.zeros(3)
            u0 = (test_stream[i].random(64)*2-1)*0.6
        rktest_u_arr_train_nonoise[:, :, i], rktest_u_arr_test[:, :, i], p, params = RungeKuttawrapped(x0=ic[0],
             y0=ic[1], z0=30*ic[2], T=rkTime, ttsplit=split, u0=u0, system=system, params=params)
        """
        print('Test data %d' % i)
        print(rktest_u_arr_test[-3:,-3:,i])
        """

    return rktest_u_arr_train_nonoise, rktest_u_arr_test, params


def test(res, noise_in, rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max, num_tests=100, rkTime=1000, split=3000, showMapError=False, showTrajectories=False, showHist=False, system='lorenz', params=np.array([[], []], dtype=np.complex128), pmap=False):
    # tic = time.perf_counter()
    stable_count, mean_sum_squared, max_sum_squared, variances, valid_time, preds, wass_dist, pmap_max, pmap_max_wass_dist = testwrapped(
        res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.Wout, res.leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test,   true_pmap_max, num_tests, rkTime, split, noise_in,  showMapError, showTrajectories, showHist, system, params=params, pmap=pmap)
    # toc = time.perf_counter()
    # runtime = toc - tic
    # print("Test " + str(i) + " valid time: " + str(j))

    return stable_count/num_tests, mean_sum_squared, max_sum_squared, variances, valid_time, preds, wass_dist, pmap_max, pmap_max_wass_dist


@jit(nopython=True, fastmath=True)
def testwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape, Wout, leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test,   true_pmap_max, num_tests, rkTime, split, noise_in, showMapError=True,   showTrajectories=True, showHist=True, system='lorenz', tau=0.1, params=np.array([[], []], dtype=np.complex128), pmap=False):
    stable_count = 0
    valid_time = np.zeros(num_tests)
    max_sum_square = np.zeros(num_tests)
    mean_sum_square = np.zeros(num_tests)
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    wass_dist = np.zeros(num_tests)
    pmap_max = []
    pmap_max_wass_dist = np.zeros(num_tests)
    preds = np.zeros((num_tests, rktest_u_arr_test.shape[0], (rkTime-split)+1))

    # print(num_tests)
    for i in range(num_tests):
        with objmode(test_tic='double'):
            test_tic = time.perf_counter()
        """
        np.random.seed(i)
        ic = np.random.rand(3)*2-1
        u_arr_train, u_arr_train_nonoise, u_arr_test, train_length, noise_scaling = \
            RungeKuttawrapped(x0 = ic[0], y0 = ic[1],
                              z0 = 30*ic[2], T = rkTime, ttsplit = split)
        """
        res_X = (np.zeros((res_X.shape[0], split+2))*2 - 1)
        # print('Win')
        # print(Win[:3,:3])
        # print('A')
        # print(W[:3,:3])
        # print('Wout')
        # print(Wout[:3,:3])

        # sets res.X
        res_X, p = getXwrapped(np.ascontiguousarray(
            rktest_u_arr_train_nonoise[:, :, i]), res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noise_in)
        pred = predictwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape,
                              Wout, leakage, u0=rktest_u_arr_test[:, 0, i], steps=(rkTime-split))
        """
        with objmode():
            print('Test %d pred:' % i)
        print(pred[-3:,-3:])
        """
        preds[i] = pred
        error = np.zeros(pred[0].size)
        if pmap:
            if system == 'lorenz':
                calc_pred = np.stack((pred[0]*7.929788629895004,
                     pred[1]*8.9932616136662, pred[2]*8.575917849311919+23.596294463016896))
                # wass_dist[i] = wasserstein_distance_empirical(calc_pred.flatten(), true_trajectory.flatten())
                pred_pmap_max = poincare_max(calc_pred, np.arange(pred.shape[0]))
            elif system == 'KS':
                # wass_dist[i] = wasserstein_distance_empirical(pred.flatten()*1.1876770355823614, true_trajectory.flatten())
                 pred_pmap_max = poincare_max(
                    pred*1.1876770355823614, np.arange(pred.shape[0]))

            pmap_max.append(pred_pmap_max)
            for j in range(rktest_u_arr_test.shape[0]):
                if j == 0:
                    pred_pmap_max_all = pred_pmap_max[j]
                else:
                    pred_pmap_max_all = np.append(
                        pred_pmap_max_all, pred_pmap_max[j])
            pmap_max_wass_dist[i] = wasserstein_distance_empirical(pred_pmap_max_all, true_pmap_max)
        else:
            pmap_max_wass_dist[i] = np.nan
            #placeholder = [np.nan]
            #pmap_max.append(placeholder)

        # print(pred.size)

        """
        plt.pcolor(pred, vmin = -3, vmax = 3)
        plt.show()

        plt.pcolor(rktest_u_arr_test[:,:,i], vmin = -3, vmax = 3)
        plt.show()
        """

        vt_cutoff = 0.2*np.sqrt(2)
        check_vt = True
        array_compute = True
        for j in range(1, pred[0].size):
            error[j] = np.sqrt(
                np.mean((pred[:, j]-rktest_u_arr_test[:, j, i])**2.0))

            if error[j] < vt_cutoff and check_vt:
                valid_time[i] = j
            else:
                check_vt = False
        if array_compute:
            if system == 'lorenz':
                rkmap_u_arr_train = RungeKuttawrapped_pred(u0_array=np.stack((pred[0]*7.929788629895004,
                    pred[1]*8.9932616136662, pred[2]*8.575917849311919+23.596294463016896)),
                    h=0.01, system=system, params=params, tau=tau, ttsplit=pred.shape[1])[0]
            elif system == 'KS':
                u0 = pred*1.1876770355823614
                rkmap_u_arr_train = RungeKuttawrapped_pred(
                    u0_array=u0, h=tau, T=1, system=system, params=params, ttsplit=pred.shape[1])[0]
            # print(rkmap_u_arr_train[0,:10])
            x2y2z2 = sum_numba_axis0(
                (pred[:, 1:]-rkmap_u_arr_train[:, :-1])**2.0)
        else:
            x2y2z2 = np.zeros(pred[0].size-1)
            for j in range(1, pred[0].size):

                if system == 'lorenz':
                    rkmap_u_arr_train = RungeKuttawrapped(pred[0][j-1]*7.929788629895004, pred[1][j-1]*8.9932616136662, pred[2]
                                                          [j-1]*8.575917849311919+23.596294463016896, h=0.01, T=1, tau=tau, system=system, params=params)[0]
                elif system == 'KS':
                    u0 = pred[:, j-1]*(1.1876770355823614)
                    rkmap_u_arr_train = RungeKuttawrapped(
                        0, 0, 0, h=tau, T=1, u0=u0, system=system, params=params)[0]
                # if j <= 10:
                # print(rkmap_u_arr_train[0,1])

                x2y2z2[j-1] = np.sum((pred[:, j]-rkmap_u_arr_train[:, 1])**2)
        x2y2z2 = x2y2z2/np.sqrt(2.0)
        # print("Mean: " + str(np.mean(pred[0])))
        # print("Variance: " + str(np.var(pred[0])))
        """
        if showHist:
            plt.figure()
            plt.hist(pred[0], bins = 11, label = "Predictions", alpha = 0.75)
            plt.hist(rktest_u_arr_test[0,:,i],
                     bins = 11, label = "Truth", alpha = 0.75)
            plt.legend(loc="upper right")

        if showMapError:
            # plt.figure()
            # plt.plot(vector_field, label = "Vector Field Stability Metric")
            # plt.legend(loc="upper right")

            plt.figure()
            plt.plot(x2y2z2, label = "x + y + z square error")
            plt.legend(loc="upper right")

        if showTrajectories:
            plt.figure()
            plt.plot(pred[0], label = "Predictions")
            plt.plot(rktest_u_arr_test[0,:,i], label = "Truth")
            plt.ylim(-3,3)
            plt.legend(loc="upper right")

        print("Variance of lorenz data x dim: " + \
              str(np.var(rktest_u_arr_test[0,:,i])))
        print("Variance of predictions: " + str(np.var(pred[0])))
        print("Max of total square error: " + str(np.max(x2y2z2)))
        print("Mean of total error: " + str(np.mean(x2y2z2)))
        print("Wasserstein distance: " + \
              str(wasserstein_distance(pred[0], rktest_u_arr_test[0,:,i])))
        print()
        """

        max_sum_square[i] = np.max(x2y2z2)
        mean_sum_square[i] = np.mean(x2y2z2)
        # print(mean_sum_square)
        if system == 'lorenz':
            means[i] = np.mean(pred[0])
            variances[i] = np.var(pred[0])
        elif system == 'KS':
            means[i] = np.mean(pred.flatten())
            variances[i] = np.var(pred.flatten())

        # print('Map error: ', mean_sum_square[i])
        # print('Variance: ', variances[i])
        # print('True Variance: ', np.var(rktest_u_arr_test))
        if mean_sum_square[i] < 5e-3 and 0.9 < variances[i] and variances[i] < 1.1:
            stable_count += 1
            # print("stable")
            # print()
        # else:
            # print("unstable")
            # print()
        with objmode(test_toc='double'):
            test_toc = time.perf_counter()
        test_time = test_toc - test_tic
        # print(test_time)

    """
    if showMapError or showTrajectories or showHist:
        plt.show()

    # print("Variance of total square error: " + str(np.var(x2y2z2)))

    print("Avg. max sum square: " + str(np.mean(max_sum_square)))
    print("Avg. mean sum square: " + str(np.mean(mean_sum_square)))
    print("Avg. of x dim: " + str(np.mean(means)))
    print("Var. of x dim: " + str(np.mean(variances)))
    print()
    """

    return stable_count, mean_sum_square, max_sum_square, variances, valid_time, preds, wass_dist, pmap_max, pmap_max_wass_dist


def generate_res(res_gen, rk, res_size, rho, sigma, leakage, win_type, bias_type, noise_stream, noisetype='none', noise_scaling=0, noise_realizations=1, traintype='normal', skip=150):
    reservoir = Reservoir(rk, res_gen, rk.u_arr_train.shape[0], rsvr_size=res_size,
                spectral_radius=rho, input_weight=sigma, leakage=leakage, win_type=win_type, bias_type=bias_type)
    # print('Train Data shape: (%d, %d)' % (rk.u_arr_train.shape[0], rk.u_arr_train.shape[1]))
    # print(rk.u_arr_train[-3:,-3:])
    data_shape = rk.u_arr_train.shape
    res_shape = reservoir.X.shape
    noise_in = gen_noise_driver(data_shape, res_shape, traintype,
                                noisetype, noise_scaling, noise_stream, noise_realizations)
    print(noise_in.shape)
    get_states(reservoir, rk, noise_in, noisetype, noise_scaling,
               noise_realizations, traintype, skip)
    return reservoir, noise_in


def optim_func(res, noise_in, noise, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha,   true_pmap_max, rkTime=400, split=2000, traintype='normal', system='lorenz', params=np.array([[], []], dtype=np.complex128), pmap = False):

    # try:
    idenmat = np.identity(
        res.rsvr_size+1+rktest_u_arr_train_nonoise.shape[0])*alpha
    if traintype not in ['sylvester', 'sylvester_wD']:
        # print('Noise mag: %e' % noise)
        # print('Gradient reg:')
        # print(res.gradient_reg[:5,:5])
        res.Wout = np.transpose(solve(np.transpose(
            res.states_trstates + noise**2.0*res.gradient_reg+idenmat), np.transpose(res.data_trstates)))
    else:
        res.Wout = solve_sylvester(
            res.left_mat, res.states_trstates+idenmat, res.data_trstates)
    out = test(res, noise_in, rktest_u_arr_train_nonoise, rktest_u_arr_test,   true_pmap_max, num_tests=num_tests, rkTime=rkTime, split=split,
        showMapError=True, showTrajectories=True, showHist=True, system=system, params=params, pmap = pmap)
    results = out[0]
    variances = out[3]
    mean_sum_squared = out[1]
    max_sum_squared = out[2]
    valid_time = out[4]
    preds = out[5]
    wass_dist = out[6]
    pmap_max = out[7]
    pmap_max_wass_dist = out[8]
    # except:
        # print("eigenvalue error occured.")

    return -1*results, mean_sum_squared, max_sum_squared, variances, valid_time, preds, wass_dist, pmap_max, pmap_max_wass_dist


def get_res_results(itr, res_gen, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, noise_stream, traintype,
    rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha_values, rkTime_test, split_test, system, tau, params, savepred, debug_mode, train_seed,   true_pmap_max, pmap):
    tic = time.perf_counter()
    print('Starting res %d' % itr)
    reservoir, noise_in = generate_res(res_gen, rk, res_size, rho, sigma, leakage,
                                       win_type, bias_type, noise_stream, noisetype, noise, noise_realizations, traintype)

    toc = time.perf_counter()
    print('Res states found for itr %d, runtime: %f sec.' % (itr, toc-tic))
    # for r in reservoirs:
    # r.data_trstates = 0
    # r.states_trstates = 0
    #    get_states(r, rk, skip = 150)

    final_out = []
    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise
    for noise in noise_array:
        stable_frac = np.zeros((alpha_values.size-1))
        mean_sum_squared = np.zeros((num_tests, alpha_values.size-1))
        max_sum_squared = np.zeros((num_tests, alpha_values.size-1))
        variances = np.zeros((num_tests, alpha_values.size-1))
        valid_time = np.zeros((num_tests, alpha_values.size-1))
        wass_dist = np.zeros((num_tests, alpha_values.size-1))
        # pmap_max           = []
        pmap_max_wass_dist = np.zeros((num_tests, alpha_values.size-1))
        noise_tic = time.perf_counter()
        min_optim_func = lambda alpha: optim_func(reservoir, noise_in[0], noise, rktest_u_arr_train_nonoise, rktest_u_arr_test,
                                                  num_tests, alpha,   true_pmap_max, rkTime_test, split_test, traintype, system, params, pmap)
        func_vals = np.zeros(alpha_values.size)
        for j in range(alpha_values.size):
            print('Regularization: ', alpha_values[j])
            if debug_mode:
                out = min_optim_func(alpha_values[j])
                if j == 0:
                    stable_frac_0 = out[0]
                    variances_0 = out[3]
                    mean_sum_squared_0 = out[1]
                    max_sum_squared_0 = out[2]
                    valid_time_0 = out[4]
                    wass_dist_0 = out[6]
                else:
                    stable_frac[j-1] = out[0]
                    variances[:, j-1] = out[3]
                    mean_sum_squared[:, j-1] = out[1]
                    max_sum_squared[:,j-1] = out[2]
                    valid_time[:, j-1] = out[4]
                    wass_dist[:, j-1] = out[6]
                    # pmap_max.append(out[7])
                    pmap_max_wass_dist[:, j-1] = out[8]
                if savepred:
                    if j == 1:
                        preds = out[5]
                    elif j == 2:
                        preds = np.stack((preds, out[5]), axis=3)
                    elif j > 2:
                        preds = np.concatenate((preds, out[5].reshape(
                            out[5].shape[0], out[5].shape[1], out[5].shape[2], 1)), axis=3)
            else:
                try:
                    out = min_optim_func(alpha_values[j])
                    if j == 0:
                        stable_frac_0 = out[0]
                        variances_0 = out[3]
                        mean_sum_squared_0 = out[1]
                        max_sum_squared_0 = out[2]
                        valid_time_0 = out[4]
                        wass_dist_0 = out[6]
                    else:
                        stable_frac[j-1] = out[0]
                        variances[:, j-1] = out[3]
                        mean_sum_squared[:, j-1] = out[1]
                        max_sum_squared[:, j-1] = out[2]
                        valid_time[:, j-1] = out[4]
                        wass_dist[:, j-1] = out[6]
                        # pmap_max.append(out[7])
                        pmap_max_wass_dist[:, j-1] = out[8]
                    if savepred:
                        if j == 1:
                            preds = out[5]
                        elif j == 2:
                            preds = np.stack((preds, out[5]), axis=3)
                        elif j > 2:
                            preds = np.concatenate((preds, out[5].reshape(
                                out[5].shape[0], out[5].shape[1], out[5].shape[2], 1)), axis=3)
                except:
                    print('Training unsucessful for alpha:')
                    print(alpha_values[j])
                    if j == 0:
                        stable_frac_0 = 0.0
                        variances_0 = np.zeros(num_tests)
                        mean_sum_squared_0 = np.zeros(num_tests)
                        max_sum_squaredi_0 = np.zeros(num_tests)
                        valid_time_0 = np.zeros(num_tests)
                        wass_dist_0 = np.zeros(num_tests)
                    else:
                        stable_frac[j-1] = 0.0
                        variances[:, j-1] = np.zeros(num_tests)
                        mean_sum_squared[:, j-1] = np.zeros(num_tests)
                        max_sum_squared[:, j-1] = np.zeros(num_tests)
                        valid_time[:, j-1] = np.zeros(num_tests)
                        wass_dist[:, j-1] = np.zeros(num_tests)
                        # pmap_max.append(np.empty((2,2), dtype = np.float64))
                        pmap_max_wass_dist[:, j-1] = np.zeros(num_tests)
                    if savepred:
                        pred_out = np.zeros(
                            (num_tests, rktest_u_arr_test.shape[0], (rkTime_test-split_test)+1))
                        if j == 1:
                            preds = preds_out
                        elif j == 2:
                            preds = np.stack((preds, preds_out), axis=3)
                        elif j > 2:
                            preds = np.concatenate((preds, preds_out.reshape(
                                preds_out.shape[0], preds_out.shape[1], preds_out.shape[2], 1)), axis=3)
        if not savepred:
            preds = np.empty((1, 1, 1, 1), dtype=np.complex128)

        final_out.append((np.copy(stable_frac_0), np.copy(stable_frac), np.copy(mean_sum_squared_0), np.copy(mean_sum_squared), np.copy(max_sum_squared_0), np.copy(max_sum_squared), np.copy(variances_0), np.copy(variances),  np.copy(valid_time_0), np.copy(valid_time), np.copy(preds), train_seed, noise, itr, np.copy(wass_dist_0), np.copy(wass_dist), np.copy(pmap_max_wass_dist)))  # pmap_max, np.copy(pmap_max_wass_dist)))
        noise_toc = time.perf_counter()
        print('Noise test time: %f sec.' % (noise_toc - noise_tic))
    toc = time.perf_counter()
    runtime = toc - tic
    print('Iteration runtime: %f sec.' % runtime)
    return final_out


def find_stability(noisetype, noise, traintype, train_seed, train_gen, res_itr, res_gen, test_stream, noise_stream, rho, sigma, leakage, win_type, bias_type, train_time, res_size, res_per_test, noise_realizations, num_tests, alpha_values, system, tau, savepred, debug_mode, foldername, pmap):
    import warnings
    warnings.filterwarnings("ignore")
    if isinstance(noise, np.ndarray):
        print_str = 'Noise: ['
        for i in range(noise.size):
            print_str += '%e, ' % noise[i]
        print_str += ']'
        print(print_str)
    else:
        print('Noise: %e' % noise)
    print('Training seed: %d' % train_seed)
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    if system == 'lorenz':
        rkTime_test = 4000
        split_test = 2000
    elif system == 'KS':
        time_mult = 0.25/tau
        rkTime_test = int(18000*time_mult)
        split_test = int(2000*time_mult)

        # rkTime_test = 3000
    # split_test  = 2000
    rktest_u_arr_train_nonoise, rktest_u_arr_test, params = get_test_data(
        test_stream, tau=tau, num_tests=num_tests, rkTime=rkTime_test, split=split_test, system=system)
    # np.random.seed(train_seed)
    if system == 'lorenz':
        ic = train_gen.random(3)*2-1
        rk = RungeKutta(x0=ic[0], y0=ic[1], z0=30*ic[2], tau=tau,
                        T=train_time, ttsplit=train_time, system=system, params=params)
    elif system == 'KS':
        u0 = 0.6*(train_gen.random(64)*2-1)
        rk = RungeKutta(0, 0, 0, tau=tau, T=train_time,
                        ttsplit=train_time, u0=u0, system=system, params=params)
    #print('Training data %d:' % train_seed)
    #print(rk.u_arr_train[-3:,-3:])
    #true_filename = foldername + \
    #    '%s_tau%0.2f_true_trajectory.csv' % (system, tau)
    true_pmap_max_filename = foldername + \
        '%s_tau%0.2f_true_pmap_max.csv' % (system, tau)
    #true_trajectory = np.loadtxt(true_filename, delimiter=',')
    if pmap:
        true_pmap_max = np.loadtxt(true_pmap_max_filename, delimiter=',')
        print('Snippet of true poincare map:')
        print(true_pmap_max[:5])
    else:
        true_pmap_max = np.zeros(100)

    out = get_res_results(res_itr, res_gen, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, noise_stream, traintype, rktest_u_arr_train_nonoise,
                          rktest_u_arr_test, num_tests, alpha_values, rkTime_test, split_test, system, tau, params, savepred, debug_mode, train_seed,   true_pmap_max, pmap)
    # toc_global = time.perf_counter()
    # print('Total Runtime: %s sec.' % (toc_global - tic_global))

    return out

@ray.remote
def find_stability_remote(*args):
    return find_stability(*args)

def find_stability_serial(*args):
    return find_stability(*args)


def get_stability_output(out_full, noise_indices, train_indices, res_per_test, num_tests, alpha_values, savepred, metric = 'mss_var', return_all = False):#metric='pmap_max_wass_dist'):
    noise_vals = np.unique(noise_indices)
    train_vals = np.unique(train_indices)
    """
    print(train_vals)
    print(noise_vals)
    """
    tn, nt = np.meshgrid(train_vals, noise_vals)
    tn = tn.flatten()
    nt = nt.flatten()
    results = []

    stable_frac_0 = np.zeros((train_vals.size, noise_vals.size, res_per_test))
    stable_frac = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, alpha_values.size-1))
    mean_sum_squared_0 = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests))
    mean_sum_squared = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests, alpha_values.size-1))
    max_sum_squared_0 = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests))
    max_sum_squared = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests, alpha_values.size-1))
    variances_0 = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests))
    variances = np.zeros((train_vals.size, noise_vals.size,
                         res_per_test, num_tests, alpha_values.size-1))
    valid_time_0 = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests))
    valid_time = np.zeros((train_vals.size, noise_vals.size,
                          res_per_test, num_tests, alpha_values.size-1))
    wass_dist_0 = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests))
    wass_dist = np.zeros((train_vals.size, noise_vals.size,
                         res_per_test, num_tests, alpha_values.size-1))
    # pmap_max          = np.zeros((train_vals.size, noise_vals.size, res_per_test), dtype=object)
    pmap_max_wass_dist = np.zeros(
        (train_vals.size, noise_vals.size, res_per_test, num_tests, alpha_values.size-1))

    for k, train in enumerate(train_vals):
        for j, noise in enumerate(noise_vals):
            out = out_full[(train == train_indices) & (noise == noise_indices)]
            for i in range(res_per_test):
                stable_frac_0[k, j, i] = out[i][0]
                stable_frac[k, j, i, :] = out[i][1]
                mean_sum_squared_0[k, j, i] = out[i][2]
                mean_sum_squared[k, j, i, :, :] = out[i][3]
                max_sum_squared_0[k, j, i] = out[i][4]
                max_sum_squared[k, j, i, :, :] = out[i][5]
                variances_0[k, j, i] = out[i][6]
                variances[k, j, i, :, :] = out[i][7]
                valid_time_0[k, j, i] = out[i][8]
                valid_time[k, j, i, :, :] = out[i][9]
                wass_dist_0[k, j, i] = out[i][14]
                wass_dist[k, j, i, :, :] = out[i][15]
                pmap_max_wass_dist[k, j, i, :, :] = out[i][16]
                # pmap_max[k,j,i] = out[i][14]
    if return_all and not savepred:
        for k in range(tn.size):
            train_idx = np.where(tn[k] == train_vals)[0][0]
            noise_idx = np.where(nt[k] == noise_vals)[0][0]
            result = (stable_frac_0[train_idx, noise_idx], stable_frac[train_idx, noise_idx],\
                mean_sum_squared_0[train_idx, noise_idx],\
                mean_sum_squared[train_idx,noise_idx], max_sum_squared_0[train_idx, noise_idx],\
                max_sum_squared[train_idx,noise_idx], variances_0[train_idx, noise_idx],\
                variances[train_idx,noise_idx], valid_time_0[train_idx, noise_idx],\
                valid_time[train_idx,noise_idx])

            results.append(result)
        return results
    elif return_all and savepred:
        raise ValueError
    else:
        best_stable_frac = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test))
        best_mean_sum_squared = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        best_max_sum_squared = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        best_variances = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        best_valid_time = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        best_wass_dist = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        # best_pmap_max   = np.zeros((train_vals.size, noise_vals.size, res_per_test, num_tests), dtype = object)
        best_pmap_max_wass_dist = np.zeros(
            (train_vals.size, noise_vals.size, res_per_test, num_tests))
        stable_frac_alpha = np.zeros(noise_vals.size)
        best_j = np.zeros(noise_vals.size)
        for i, noise in enumerate(noise_vals):
            if metric == 'mss_var':
                best_alpha_val = 0
            elif metric in ['wass_dist', 'pmap_max_wass_dist', 'mean_sum_squared', 'max_sum_squared']:
                best_alpha_val = np.inf
            for j in range(1, alpha_values.size):
                if metric == 'mss_var':
                    metric_flag = np.mean(
                        stable_frac[:, i, :, j-1]) <= best_alpha_val
                elif metric == 'wass_dist':
                    metric_flag = np.mean(
                        wass_dist[:, i, :, :, j-1]) <= best_alpha_val
                elif metric == 'pmap_max_wass_dist':
                    #print(j)
                    #print(np.mean(pmap_max_wass_dist[:, i, :, :, j-1]))
                    metric_flag = np.mean(
                        pmap_max_wass_dist[:, i, :, :, j-1]) <= best_alpha_val
                elif metric == 'mean_sum_squared':
                    metric_flag = np.mean(mean_sum_squared[:,i,:,:,j-1]) <= best_alpha_val
                elif metric == 'max_sum_squared':
                    metric_flag = np.median(max_sum_squared[:,i,:,:,j-1]) <= best_alpha_val
                if metric_flag or (metric == 'mss_var' and best_alpha_val == 0 and j == alpha_values.size-1) or \
                        (metric in ['wass_dist', 'pmap_max_wass_dist', 'mean_sum_squared', 'max_sum_squared'] \
                        and np.isinf(best_alpha_val) and j == alpha_values.size-1):
                    if metric == 'mss_var':
                        best_alpha_val = np.mean(stable_frac[:, i, :, j-1])
                    elif metric == 'wass_dist':
                        best_alpha_val = np.mean(wass_dist[:, i, :, :, j-1])
                    elif metric == 'pmap_max_wass_dist':
                        best_alpha_val = np.mean(
                            pmap_max_wass_dist[:, i, :, :, j-1])
                    elif metric == 'mean_sum_squared':
                        best_alpha_val = np.mean(mean_sum_squared[:,i,:,:,j-1])
                    elif metric == 'max_sum_squared':
                        best_alpha_val = np.median(max_sum_squared[:,i,:,:,j-1])
                    best_stable_frac[:, i] = -stable_frac[:, i, :, j-1]
                    best_variances[:, i] = variances[:, i, :, :, j-1]
                    best_mean_sum_squared[:, i] = mean_sum_squared[:, i, :, :, j-1]
                    best_max_sum_squared[:, i] = max_sum_squared[:, i, :, :, j-1]
                    best_valid_time[:, i] = valid_time[:, i, :, :, j-1]
                    best_wass_dist[:, i] = wass_dist[:, i, :, :, j-1]
                    best_pmap_max_wass_dist[:,
                        i] = pmap_max_wass_dist[:, i, :, :, j-1]
                    stable_frac_alpha[i] = alpha_values[j]
                    best_j[i] = int(j)
                    # for k,l,m in product(np.arange(train_vals.size), np.arange(res_per_test), np.arange(num_tests)):
                    #    best_pmap_max[k,i,l,m] = pmap_max[k,i,l][j-1][m]
        print(best_j)
        for k in range(tn.size):
            if savepred:
                out = out_full[(tn[k] == train_indices) & (nt[k] == noise_indices)]
                for i in range(res_per_test):
                    if i == 0:
                        preds = out[i][10]
                    elif i == 1:
                        preds = np.stack((preds, out[i][10]), axis=0)
                    elif i > 1:
                        preds = np.concatenate((preds, out[i][10].reshape(
                            1, out[i][10].shape[0], out[i][10].shape[1], out[i][10].shape[2], out[i][10].shape[3])), axis=0)
                if res_per_test == 1:
                    preds = preds.reshape(
                        1, preds.shape[0], preds.shape[1], preds.shape[2], preds.shape[3])
                # print(nt[k])
                # print(noise_vals)
                # print(nt[k] == noise_vals)
                noise_idx = np.where(nt[k] == noise_vals)[0][0]

                # print(preds.shape)
                # print(noise_idx)
                best_preds = preds[:, :, :, :, int(best_j[noise_idx])-1]
                """
                print('Training index: %d' % tn[k])
                print('Noise index: %e' % nt[k])
                print('End of best pred:')
                print(best_preds[:,:,-1,-1])
                """
            else:
                best_preds = np.empty((1, 1, 1, 1), dtype=np.complex128)

            train_idx = np.where(tn[k] == train_vals)[0][0]
            noise_idx = np.where(nt[k] == noise_vals)[0][0]
            result = (stable_frac_0[train_idx, noise_idx], best_stable_frac[train_idx, noise_idx],
                stable_frac_alpha[noise_idx], mean_sum_squared_0[train_idx, noise_idx],
                best_mean_sum_squared[train_idx,
                    noise_idx], max_sum_squared_0[train_idx, noise_idx],
                best_max_sum_squared[train_idx,
                    noise_idx], variances_0[train_idx, noise_idx],
                best_variances[train_idx,
                    noise_idx], valid_time_0[train_idx, noise_idx],
                best_valid_time[train_idx,
                    noise_idx], np.copy(best_preds), wass_dist_0[train_idx, noise_idx],
                best_wass_dist[train_idx, noise_idx], best_pmap_max_wass_dist[train_idx, noise_idx])  # pmap_max[train_idx, noise_idx], best_pmap_max_wass_dist[train_idx, noise_idx])

            results.append(result)

        return results


def main(argv):
    print('Function Started')
    train_time = 3000
    res_size = 1000
    res_per_test = 20
    noise_realizations = 1
    num_tests = 10
    num_trains = 25
    traintype = 'normal'
    noisetype = 'gaussian'
    system = 'KS'
    savepred = False
    rho = 0.5
    sigma = 1.0
    leakage = 1.0
    bias_type = 'old'
    win_type = 'full'
    debug_mode = False
    pmap = False
    machine = 'deepthought2'
    ifray = True
    tau_flag = True
    num_cpus = 20
    metric = 'mss_var'
    return_all = False

    try:
        opts, args = getopt.getopt(argv, "T:N:r:",
                ['noisetype=', 'traintype=', 'system=', 'res=',
                'tests=', 'trains=', 'savepred=', 'tau=', 'rho=',
                'sigma=', 'leakage=', 'bias_type=', 'debug=', 'win_type=',
                'machine=', 'num_cpus=', 'pmap=', 'parallel=', 'metric=','returnall='])
    except getopt.GetoptError:
        print('Error: Some options not recognized')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-T':
            train_time = int(arg)
            print('Training iterations: %d' % train_time)
        elif opt == '-N':
            res_size = int(arg)
            print('Reservoir nodes: %d' % res_size)
        elif opt == '-r':
            noise_realizations = int(arg)
            print('Noise Realizations: %d' % noise_realizations)
        elif opt == '--metric':
            metric = str(arg)
            if metric not in ['wass_dist', 'pmap_max_wass_dist', 'mean_sum_squared', 'max_sum_squared', 'mss_var']:
                raise ValueError
            print('Stability metric: %s' % metric)
        elif opt == '--returnall':
            if arg == 'True':
                return_all = True
            elif arg == 'False':
                return_all = False
            else:
                raise ValueError
        elif opt == '--parallel':
            parallel_in = str(arg)
            if parallel_in == 'True':
                ifray = True
            elif parallel_in == 'False':
                ifray = False
            else:
                raise ValueError
        elif opt == '--pmap':
            pmap_in = str(arg)
            if pmap_in == 'True':
                pmap = True
            elif pmap_in == 'False':
                pmap = False
            else:
                raise ValueError
        elif opt == '--machine':
            machine = str(arg)
            if not machine == 'skynet' and not machine == 'deepthought2':
                raise ValueError
            print('Machine: %s' % machine)
        elif opt == '--num_cpus':
            num_cpus = int(arg)
            print('Number of CPUS: %d' % num_cpus)
        elif opt == '--rho':
            rho = float(arg)
            print('Rho: %f' % rho)
        elif opt == '--sigma':
            sigma = float(arg)
            print('Sigma: %f' % sigma)
        elif opt == '--leakage':
            leakage = float(arg)
            print('Leakage: %f' % leakage)
        elif opt == '--tau':
            tau  = float(arg)
            tau_flag = False
            print('Reservoir timestep: %f' % tau)
        elif opt == '--win_type':
            win_type = str(arg)
            print('Win Type: %s' % win_type)
        elif opt == '--bias_type':
            bias_type = str(arg)
            print('Bias Type: %s' % bias_type)
        elif opt == '--res':
            res_per_test = int(arg)
            print('Number of reservoirs: %d' % res_per_test)
        elif opt == '--tests':
            num_tests = int(arg)
            print('Number of tests: %d' % num_tests)
        elif opt == '--trains':
            num_trains = int(arg)
            print('Number of training data sequences: %d' % num_trains)
        elif opt == '--savepred':
            if arg == 'True':
                savepred = True
            elif arg == 'False':
                savepred = False
            print('Saving predictions: %s' % arg)
        elif opt == '--noisetype':
            noisetype = str(arg)
            print('Noise type: %s' % noisetype)
        elif opt == '--traintype':
            traintype = str(arg)
            print('Training type: %s' % traintype)
        elif opt == '--system':
            system = str(arg)
            print('System: %s' % system)
        elif opt == '--debug':
            if arg == 'True':
                debug_mode = True
            elif arg == 'False':
                debug_mode = False
            print('Debug Mode: %s' % arg)
    if return_all and savepred:
        print('Cannot return results for all parameters and full predictions due to memory constraints.')
        raise ValueError
    if tau_flag:
        if system == 'lorenz':
            tau = 0.1
        elif system == 'KS':
            tau = 0.25
    if savepred:
        predflag = 'wpred_'
    else:
        predflag = ''
    if machine == 'skynet':
        if ifray:
            ray.init(num_cpus = num_cpus)
        foldername = '/h/awikner/res-noise-stabilization/'
    elif machine == 'deepthought2':
        if ifray:
            ray.init(address=os.environ["ip_head"])
        foldername = '/lustre/awikner1/res-noise-stabilization/'
    print(foldername)

    ########################################
    # train_time = 500
    # res_size = 100
    # res_per_test = 100
    # noise_realizations = 1
    #sys.path.append(foldername)
    #from lorenzrungekutta_numba import rungekutta, rungekutta_pred
    #from ks_etdrk4 import kursiv_predict, kursiv_predict_pred
    #from csc_mult import mult_vec, construct_jac_mat_csc, matrix_diag_mult,\
    #        matrix_dot_left_T, matrix_diag_sparse_mult_add, matrix_dot_left_T
    #from poincare_max import poincare_max


    noise_values_array = np.logspace(-4, 0, num = 13, base = 10)[5:10]
    # noise_values_array = np.array([np.logspace(-4, 0, num = 13, base = 10)[6]])
    # noise_values_array = np.array([0,1e-3,1e-2])
    alpha_values = np.append(0., np.logspace(-7, -2, 11))
    if traintype in ['normal','normalres1','normalres2','rmean','rmeanres1','rmeanres2',
            'rplusq','rplusqres1','rplusqres2']:
        alpha_values = alpha_values*noise_realizations
    # alpha_values = np.array([0,1e-6,1e-4])
    ss_res   = np.random.SeedSequence(12)
    ss_train = np.random.SeedSequence(34)
    ss_test  = np.random.SeedSequence(56)
    ss_noise = np.random.SeedSequence(78)
    if traintype in ['gradient1','gradient2','gradient12'] or 'gradientk' in traintype:
        res_seeds       = ss_res.spawn(res_per_test)
        train_seeds     = ss_train.spawn(num_trains)
        test_seeds      = ss_test.spawn(num_tests)
        res_streams     = np.zeros(res_per_test*num_trains, dtype = object)
        train_streams   = np.zeros(res_per_test*num_trains, dtype = object)
        test_streams    = np.zeros((res_per_test*num_trains, num_tests), dtype = object)
        for i in range(res_per_test*num_trains):
            test_streams[i] = np.array([np.random.default_rng(s) for s in test_seeds], dtype = object)
        noise_streams   = np.empty(noise_realizations, dtype = object)
        tr, rt = np.meshgrid(np.arange(num_trains), np.arange(res_per_test))
        tr     = tr.flatten()
        rt     = rt.flatten()
        for i in range(res_per_test*num_trains):
            res_streams[i]   = np.random.default_rng(res_seeds[rt[i]])
            train_streams[i] = np.random.default_rng(train_seeds[tr[i]])
        print('Starting Ray Computation')
        tic = time.perf_counter()
        if ifray:
            out_base  = ray.get([find_stability_remote.remote(noisetype, noise_values_array, traintype, \
                tr[i], train_streams[i], rt[i], res_streams[i], test_streams[i], noise_streams, rho, sigma, \
                leakage, win_type, bias_type, train_time, \
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, debug_mode, foldername, pmap) for i in range(tr.size)])
        else:
            out_base  = [find_stability_serial(noisetype, noise_values_array, traintype, \
                tr[i], train_streams[i], rt[i], res_streams[i], test_streams[i], noise_streams, rho, sigma, \
                leakage, win_type, bias_type, train_time, \
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, debug_mode, foldername, pmap) for i in range(tr.size)]

        # print('Ray out len: %d' % len(out_base))
        # print('Out elem len: %d' % len(out_base[0]))

        tnr, ntr, rtn = np.meshgrid(np.arange(num_trains), noise_values_array, np.arange(res_per_test))
        tnr = tnr.flatten()
        ntr = ntr.flatten()
        rtn = rtn.flatten()
        out = np.zeros(tnr.size, dtype = object)
        itr = 0
        train_i = []
        noise_i = []
        res_i   = []
        for i in range(len(out_base)):
            for j in range(len(out_base[i])):
                out[itr] = out_base[i][j]
                train_i.append(out[itr][11])
                noise_i.append(out[itr][12])
                res_i.append(out[itr][13])
                itr += 1
        final_out = []
        for train, noise, res in zip(tnr, ntr, rtn):
            idx = np.where(np.logical_and(np.logical_and(train_i == train, noise_i == noise), res_i == res))[0][0]
            final_out.append(out[idx])

        out = np.array(final_out, dtype = object)
    else:
        res_seeds       = ss_res.spawn(res_per_test)
        train_seeds     = ss_train.spawn(num_trains)
        test_seeds      = ss_test.spawn(num_tests)
        noise_seeds     = ss_noise.spawn(noise_realizations)
        res_streams     = np.zeros(num_trains*res_per_test*noise_values_array.size, dtype = object)
        train_streams   = np.zeros(num_trains*res_per_test*noise_values_array.size, dtype = object)
        test_streams    = np.zeros((num_trains*res_per_test*noise_values_array.size, num_tests), dtype = object)
        noise_streams   = np.zeros((num_trains*res_per_test*noise_values_array.size, noise_realizations), dtype = object)

        tnr, ntr, rtn = np.meshgrid(np.arange(num_trains), noise_values_array, np.arange(res_per_test))
        tnr = tnr.flatten()
        ntr = ntr.flatten()
        rtn = rtn.flatten()

        for i in range(num_trains*res_per_test*noise_values_array.size):
            res_streams[i]   = np.random.default_rng(res_seeds[rtn[i]])
            train_streams[i] = np.random.default_rng(train_seeds[tnr[i]])
            test_streams[i]  = np.array([np.random.default_rng(j) for j in test_seeds])
            noise_streams[i] = np.array([np.random.default_rng(j) for j in noise_seeds])


        print('Starting Ray Computation')
        tic = time.perf_counter()
        if ifray:
            out_base  = ray.get([find_stability_remote.remote(noisetype, ntr[i], traintype,\
                tnr[i], train_streams[i], rtn[i], res_streams[i], test_streams[i], noise_streams[i], \
                rho, sigma, leakage, win_type, bias_type, train_time, \
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, debug_mode, foldername, pmap) for i in range(tnr.size)])
        else:
            out_base  = [find_stability_serial(noisetype, ntr[i], traintype,\
                tnr[i], train_streams[i], rtn[i], res_streams[i], test_streams[i], noise_streams[i], \
                rho, sigma, leakage, win_type, bias_type, train_time, \
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, debug_mode, foldername, pmap) for i in range(tnr.size)]
        out = []
        for i in range(len(out_base)):
            for j in range(len(out_base[i])):
                out.append(out_base[i][j])
        out = np.array(out, dtype = object)

        #for k in range(tnr.size):
        #    print('Training idx: %d' % tnr[k])
        #    print('Noise val: %e' % ntr[k])
        #    print('Res idx: %d' % rtn[k])
        #    print(out[k][8][0, -1, -1])

        # out = [find_stability(noisetype, ntr[i], traintype, tnr[i], rtn[i], rho, sigma, leakage, win_type, bias_type, train_time, \
        #         res_size, res_per_test, noise_realizations, num_tests, alpha_values, system, tau, savepred, debug_mode) for i in range(tnr.size)]
    results = get_stability_output(out, ntr, tnr, res_per_test, num_tests, alpha_values, savepred, metric, return_all)

    # print(len(results[0]))
    ray.shutdown()
    toc = time.perf_counter()
    runtime = toc - tic
    print('Runtime over all cores: %f sec.' %(runtime))

    tn, nt= np.meshgrid(np.arange(num_trains), noise_values_array)
    tn = tn.flatten()
    nt = nt.flatten()

    top_folder = '%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)
    folder = '%s_more_noisetest_%srho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system,          predflag, rho, sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, noisetype, traintype)
    if not os.path.isdir(os.path.join(foldername, top_folder)):
        os.mkdir(os.path.join(foldername, top_folder))
    if not os.path.isdir(os.path.join(os.path.join(foldername, top_folder), folder)):
        os.mkdir(os.path.join(os.path.join(foldername, top_folder), folder))

    # foldername = ''
    print('Ray finished')

    if not return_all:
        stable_frac        = []
        stable_frac_0      = []
        stable_frac_alpha  = []
        mean_sum_squared   = []
        mean_sum_squared_0 = []
        max_sum_squared_0  = []
        max_sum_squared    = []
        variances          = []
        variances_0        = []
        wass_dist          = []
        wass_dist_0        = []
        valid_time_0       = []
        valid_time         = []
        mean_valid_time    = []
        # pmap_max           = []
        pmap_max_wass_dist = []
        for i in range(tn.size):
            stable_frac_0.append(results[i][0])
            stable_frac.append(results[i][1])
            stable_frac_alpha.append(results[i][2])
            mean_sum_squared_0.append(results[i][3])
            mean_sum_squared.append(results[i][4])
            max_sum_squared_0.append(results[i][5])
            max_sum_squared.append(results[i][6])
            variances_0.append(results[i][7])
            variances.append(results[i][8])
            valid_time_0.append(results[i][9])
            valid_time.append(results[i][10])
            mean_valid_time.append(np.mean(valid_time[-1]))
            wass_dist_0.append(results[i][12])
            wass_dist.append(results[i][13])
            # pmap_max.append(results[i][12])
            pmap_max_wass_dist.append(results[i][14])
        if savepred:
            preds = []
            for i in range(tn.size):
                preds.append(results[i][11])
        stable_frac = np.array(stable_frac).reshape(noise_values_array.size,-1)
        stable_frac_0 = np.array(stable_frac_0).reshape(noise_values_array.size,-1)
        stable_frac_alpha = np.array(stable_frac_alpha).reshape(noise_values_array.size,-1)
        mean_valid_time = np.array(mean_valid_time).reshape(noise_values_array.size, -1)
        np.savetxt(foldername+top_folder+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest_noreg.csv' %(res_size, train_time, noise_realizations), stable_frac_0,delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' %(res_size, train_time, noise_realizations), stable_frac, delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' %(res_size, train_time, noise_realizations), stable_frac_alpha, delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'mean_valid_time_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' %(res_size, train_time, noise_realizations), mean_valid_time, delimiter = ',')
        noise_values_array = noise_values_array.flatten()
        for i in range(nt.size):
            np.savetxt(foldername+top_folder+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), mean_sum_squared_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), mean_sum_squared[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'max_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                 %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), max_sum_squared_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'max_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                 %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), max_sum_squared[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), valid_time_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), valid_time[i], delimiter = ',')
            """
            np.savetxt(foldername+top_folder+folder+'wass_dist_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), wass_dist_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'wass_dist_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), wass_dist[i], delimiter = ',')
            """
            if pmap:
                np.savetxt(foldername+top_folder+folder+'pmap_max_wass_dist_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                    %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), pmap_max_wass_dist[i], delimiter = ',')
        if savepred:
            r_test, test_r = np.meshgrid(np.arange(0, res_per_test), np.arange(0, num_tests))
            r_test = r_test.flatten()
            test_r = test_r.flatten()
            for i in range(nt.size):
                for j in range(r_test.size):
                    print((i, r_test[j], test_r[j]))
                    np.savetxt(foldername+top_folder+folder+'pred_%dnodes_%dtrainiters_%dnoisereals_noise%e_train%d_res%d_test%d.csv' \
                        %(res_size, train_time, noise_realizations, nt[i], tn[i]+1, r_test[j]+1, test_r[j]+1),\
                        preds[i][r_test[j], test_r[j]], delimiter = ',')
        print('Results Saved')
    elif return_all:
        stable_frac        = []
        stable_frac_0      = []
        mean_sum_squared   = []
        mean_sum_squared_0 = []
        max_sum_squared_0  = []
        max_sum_squared    = []
        variances          = []
        variances_0        = []
        valid_time_0       = []
        valid_time         = []
        mean_valid_time    = []
        for i in range(tn.size):
            stable_frac_0.append(results[i][0])
            stable_frac.append(results[i][1])
            mean_sum_squared_0.append(results[i][2])
            mean_sum_squared.append(results[i][3])
            max_sum_squared_0.append(results[i][4])
            max_sum_squared.append(results[i][5])
            variances_0.append(results[i][6])
            variances.append(results[i][7])
            valid_time_0.append(results[i][8])
            valid_time.append(results[i][9])
            mean_valid_time.append(np.mean(valid_time[-1], axis = (0,1)))
        stable_frac = np.array(stable_frac).reshape(noise_values_array.size,-1)
        stable_frac_0 = np.array(stable_frac_0).reshape(noise_values_array.size,-1)
        mean_valid_time = np.array(mean_valid_time).reshape(noise_values_array.size, -1)
        for i, reg in enumerate(alpha_values[1:]):
            stable_frac_save = stable_frac[:,i::(alpha_values.size-1)]
            stable_frac_0_save = stable_frac_0
            mean_valid_time_save = mean_valid_time[:,i::(alpha_values.size-1)]
            np.savetxt(foldername+top_folder+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest_noreg.csv' %(res_size, train_time, noise_realizations),stable_frac_0_save,delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_reg%e_raytest.csv' %(res_size, train_time, noise_realizations, reg),stable_frac_save, delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'mean_valid_time_%dnodes_%dtrainiters_%dnoisereals_reg%e_raytest.csv' %(res_size, train_time, noise_realizations, reg),mean_valid_time_save, delimiter = ',')
        noise_values_array = noise_values_array.flatten()
        for i in range(nt.size):
            np.savetxt(foldername+top_folder+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                         %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), mean_sum_squared_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'max_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                          %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), max_sum_squared_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                     %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances_0[i], delimiter = ',')
            np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                     %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), valid_time_0[i], delimiter = ',')
            for j, reg in enumerate(alpha_values[1:]):
                np.savetxt(foldername+top_folder+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_reg%e.csv' \
                    %(res_size, train_time, noise_realizations, nt[i], tn[i]+1, reg), mean_sum_squared[i][:,:,j], delimiter = ',')
                np.savetxt(foldername+top_folder+folder+'max_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_reg%e.csv' \
                     %(res_size, train_time, noise_realizations, nt[i], tn[i]+1, reg), max_sum_squared[i][:,:,j], delimiter = ',')
                np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_reg%e.csv' \
                    %(res_size, train_time, noise_realizations, nt[i], tn[i]+1, reg), variances[i][:,:,j], delimiter = ',')
                np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_reg%e.csv' \
                    %(res_size, train_time, noise_realizations, nt[i], tn[i]+1, reg), valid_time[i][:,:,j], delimiter = ',')
        print('Results Saved')


if __name__ == "__main__":
    main(sys.argv[1:])

