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
import numpy as np
import pandas as pd
from scipy.linalg import solve, solve_sylvester
from scipy.sparse.linalg import eigs, svds, eigsh
from scipy.sparse import random
from matplotlib import pyplot as plt
from numba import jit, njit, objmode
from numba.experimental import jitclass
from numba.types import int32, int64, double
from numba.typed import List
import time

"""
import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(
    ["%s==%s" % (i.key, i.version) for i in installed_packages])
isray = [('ray==' in elem) for elem in installed_packages_list]
if (True in isray):
    print('Ray installed')
else:
    os.system('pip install -r -U ray')
"""

sys.path.append('/h/awikner/res-noise-stabilization/')
from helpers.lorenzrungekutta_numba import *
from helpers.ks_etdrk4 import *
from helpers.csc_mult import *
from helpers.poincare_max import *
from helpers.get_run_opts import *
#from helpers.matlab_funcs import *

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

@njit
def str_to_int(s):
    # Converts a string to an int in numba compiled functions
    final_index, result = len(s) - 1, 0
    for i, v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result


@njit(fastmath=True)
def mean_numba_axis1(mat):
    # Computes the mean over axis 1 in numba compiled functions
    res = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        res[i] = np.mean(mat[i])

    return res


@njit( fastmath=True)
def sum_numba_axis0(mat):
    # Computes the sum over axis 0 in numba compiled functions
    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.sum(mat[:, i])
    return res

@njit(fastmath = True)
def numba_var_axis0(pred):
    mean_all = sum_numba_axis0(pred)/pred.shape[0]
    variances_all = sum_numba_axis0((pred - mean_all)**2.0)/(pred.shape[0])
    return variances_all


@njit(fastmath=True)
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

@jit(fastmath = True, nopython = True)
def numba_eigsh(A):
    np.random.seed(0)
    v0 = np.random.rand(A.shape[0])
    with objmode(eigs_out = 'double[:]'):
        eigs_out = eigsh(A, k=6, v0=v0, maxiter=1e5, return_eigenvectors=False)
    return eigs_out


class Reservoir:
    def __init__(self, rk, res_gen, res_itr, input_size, rsvr_size=300, spectral_radius=0.6, input_weight=1, leakage=1.0, win_type='full', bias_type='old', avg_degree = 3):
        # Define class for storing reservoir layers generated from input parameters and an input random number generator
        self.rsvr_size = rsvr_size
        self.res_itr = res_itr
        """
        print('Spectral Radius: %0.2f' % spectral_radius)
        print('Input Weight: %0.2f' % input_weight)
        print('Leakage: %0.3f' % leakage)
        print('Win Type: %s' % win_type)
        print('Bias type: %s' % bias_type)
        """

        density = avg_degree/rsvr_size
        """
        unnormalized_W = (res_gen.random((rsvr_size, rsvr_size))*2 - 1)
        for i in range(unnormalized_W[:, 0].size):
            for j in range(unnormalized_W[0].size):
                if res_gen.random(1) > avg_degree/rsvr_size:
                    unnormalized_W[i][j] = 0

        max_eig = eigs(unnormalized_W, k=1,
                       return_eigenvectors=False, maxiter=10**5)

        W_sp = csc_matrix(np.ascontiguousarray(
            spectral_radius/np.abs(max_eig[0])*unnormalized_W))
        self.W_data, self.W_indices, self.W_indptr, self.W_shape = \
                (W_sp.data, W_sp.indices, W_sp.indptr, np.array(list(W_sp.shape)))
        """
        unnormalized_W_sp = random(rsvr_size, rsvr_size, density = density, format = 'csc', data_rvs = res_gen.random)
        max_eig = eigs(unnormalized_W_sp, k=1, return_eigenvectors = False, maxiter = 10**5, v0 = res_gen.random(rsvr_size))
        W_sp = unnormalized_W_sp*spectral_radius/np.abs(max_eig[0])
        self.W_data, self.W_indices, self.W_indptr, self.W_shape = \
            (np.ascontiguousarray(W_sp.data), np.ascontiguousarray(W_sp.indices),\
            np.ascontiguousarray(W_sp.indptr), np.array(list(W_sp.shape)))
        """
        print('Adjacency matrix section:')
        print(self.W_data[:4])
        """
        if win_type == 'dense':
            if bias_type != 'new_random':
                raise ValueError
            Win = (res_gen.random(rsvr_size*(input_size+1)).reshape(rsvr_size, input_size+1)*2-1)*input_weight
        else:
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
                #Win[:, 0] = (res_gen.random(rsvr_size)*2-1)*input_weight
                q = int(rsvr_size//input_vars.size)
                for i, var in enumerate(input_vars):
                    Win[q*i:q*(i+1), var+1] = (res_gen.random(q)*2-1)*input_weight
                leftover_nodes = rsvr_size - q*input_vars.size
                #var = input_vars[res_gen.integers(
                #    input_vars.size, size=leftover_nodes)]
                #Win[rsvr_size-leftover_nodes:, var +
                #    1] = (res_gen.random(leftover_nodes)*2-1)*input_weight
                for i in range(leftover_nodes):
                    Win[rsvr_size - leftover_nodes + i, input_vars[res_gen.integers(input_vars.size)]] = \
                        (res_gen.random()*2-1)*input_weight
                print('Win nonzero elements:')
                print(np.sum(Win.flatten() != 0.))
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

        #self.Win = np.ascontiguousarray(Win)
        Win_sp = csc_matrix(Win)
        self.Win_data, self.Win_indices, self.Win_indptr, self.Win_shape =\
                np.ascontiguousarray(Win_sp.data),\
                np.ascontiguousarray(Win_sp.indices),\
                np.ascontiguousarray(Win_sp.indptr),\
                np.array(list(Win_sp.shape))

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
        elif system == 'KS_d2175':
            u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr)/(1.2146066380280796)
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
    elif system == 'KS_d2175':
        u_arr, new_params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params)
        u_arr = np.ascontiguousarray(u_arr)/(1.2146066380280796)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    # u[ttsplit], the (ttsplit+1)st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params


@jit(nopython=True, fastmath=True)
def RungeKuttawrapped_pred(h=0.01, tau=0.1, T=300, ttsplit=5000, u0_array=np.array([[], []], dtype=np.complex128), system='lorenz', params=np.array([[], []], dtype=np.complex128)):
    # Numba function for obtaining training and testing dynamical system time series data for a set of initial conditions.
    # This is used during test to compute the map error instead of a for loop over the entire prediction period.
    if system == 'lorenz':
        int_step = int(tau/h)
        u_arr = np.ascontiguousarray(
            rungekutta_pred(u0_array, h, tau, int_step))

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict_pred(
            u0_array, tau=tau, T=T, params=params)
        u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
    elif system == 'KS_d2175':
        u_arr, new_params = kursiv_predict_pred(u0_array, tau=tau, T=T, params=params, d=21.75)
        u_arr = np.ascontiguousarray(u_arr)/(1.2146066380280796)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    # u[ttsplit], the (ttsplit+1)st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params

def getX(res, rk, x0=1, y0=1, z0=1):
    # Function to obtain reservoir states when in python interpreter. Calls getXwrapped.
    u_training = rk.u_arr_train
    res.X = getXwrapped(np.ascontiguousarray(u_training), res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape,
                        res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage)

    return res.X

@jit(nopython=True, fastmath=True)
def getXwrapped(u_training, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise, noisetype='none', noise_scaling=0, noise_realization=0, traintype='normal'):
    # Numba compatible function for obtaining reservoir states using various types of noise.
    # Generally returns an array of reservoir states and the noiseless training data used as input.
    # If traintype is gradient, this function instead returns the resevoir states and the reservoir states derivatives.

    if noisetype in ['gaussian', 'perturbation']:
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq'] or 'confined' in traintype:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]+noise[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]+noise[:, i]))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0-leakage)*res_X[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i])+noise[:, i])
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
            res_X_nonoise[:, i+1] = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        if traintype in ['normal', 'rmean', 'rplusq','rmeanq','rqmean','rq']:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]+noise[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i+k]+noise[:, i+k]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                res_X[:, i+noise_steps] = temp_x
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i]))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i+k]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i+k]))
                res_X[:, i+noise_steps] = temp_x
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i])
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i+k]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i+k])
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
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]+noise[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                temp_x = np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i]))+mult_vec(
                    W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1','rmeanqres1','rqmeanres1','rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:, i]))
                temp_x = np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i]))+mult_vec(
                    W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2','rmeanqres2','rqmeanres2','rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:, i])
                temp_x = np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i]))+mult_vec(
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
            res_X_nonoise[:, i+1] = (1.0-leakage)*res_X_nonoise[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        for i in range(0, u_training[0].size-noise_steps):
            temp_x = res_X_nonoise[:, i]
            for k in range(noise_steps):
                if k == 0:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i+k]+noise[:, i+k]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                else:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i+k]))+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            res_X[:, i+noise_steps] = temp_x
        u_training_wnoise = u_training+noise
        return res_X, u_training_wnoise

    elif traintype in ['sylvester_wD'] or 'gradient' in traintype:
        rsvr_size = res_X.shape[0]
        res_D = np.zeros((rsvr_size, u_training.shape[1]+1))
        for i in range(0, u_training[0].size):
            res_internal = mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i]))+mult_vec(
                W_data, W_indices, W_indptr, W_shape, res_X[:, i])
            res_X[:, i+1] = (1.0 - leakage)*res_X[:, i] + \
                             leakage*np.tanh(res_internal)
            res_D[:, i+1] = leakage/(np.power(np.cosh(res_internal), 2.0))

        return res_X, res_D
    else:
        for i in range(0, u_training[0].size):
            res_X[:, i+1] = (1.0 - leakage)*res_X[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                1., u_training[:, i]))+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]))
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
    # Generates an array of noise states for a given noise type, number of noise realizations, and array of random number generators
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

def gen_noise(noise_size, noise_length, noisetype, noise_scaling, noise_stream, noise_realizations):
    # Generates an array of noise vectors
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


def get_states(res, squarenodes, rk, reg_train_times, noise, noisetype='none', noise_scaling=0, noise_realizations=1,
        traintype='normal', skip=500):
    # Obtains the matrices used to train the reservoir using either linear regression or a sylvester equation
    # (in the case of Sylvester or Sylvester_wD training types)
    # Calls the numba compatible wrapped function
    if traintype == 'getD':
        Dn = getD(np.ascontiguousarray(rk.u_arr_train), res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype, squarenodes)
        return Dn
    elif traintype in ['sylvester', 'sylvester_wD']:
        res.data_trstates, res.states_trstates, res.Y_train, res.X_train, res.left_mat= get_states_wrapped(
            np.ascontiguousarray(rk.u_arr_train), reg_train_times, res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype, squarenodes)
    else:
        res.data_trstates, res.states_trstates, res.Y_train, res.X_train, res.gradient_reg= get_states_wrapped(
            np.ascontiguousarray(rk.u_arr_train), reg_train_times, res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            skip, noise, noisetype, noise_scaling, noise_realizations, traintype, squarenodes)

@jit(nopython=True, fastmath = True)
def get_squared(X, rsvr_size, squarenodes, dim = 0):
    X_aug = np.copy(X)
    if not squarenodes:
        return X_aug
    else:
        X_out = np.vstack((X_aug[0].reshape(1,-1), X_aug[1:rsvr_size+1], X_aug[1:rsvr_size+1]**2.0, X_aug[rsvr_size+1:]))
        return X_out

@jit(nopython=True, fastmath = True)
def get_squared_vec(X, rsvr_size, squarenodes):
    X_aug = np.copy(X)
    if not squarenodes:
        return X_aug
    else:
        X_out = np.concatenate((np.array([X_aug[0]]), X_aug[1:rsvr_size+1],X_aug[1:rsvr_size+1]**2.0, X_aug[rsvr_size+1:]))
        return X_out


@jit(nopython=True, fastmath=True)
def get_states_wrapped(u_arr_train, reg_train_times, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, skip, noise, noisetype='none',
        noise_scaling=0, noise_realizations=1, traintype='normal', squarenodes = False, q=0):
    # Numba compatible function to obtain the matrices used to train the reservoir using either linear regression or a sylvester equation.
    # The type of matrices depends on the traintype, number of noise realizations, and noisetype
    res_X = np.ascontiguousarray(res_X)
    u_arr_train = np.ascontiguousarray(u_arr_train)
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    if squarenodes:
        res_feature_size = 2*rsvr_size
    else:
        res_feature_size = rsvr_size
    data_trstates   = np.zeros((n, res_feature_size+n+1))
    states_trstates = np.zeros((reg_train_times.size, res_feature_size+n+1, res_feature_size+n+1))
    gradient_reg    = np.zeros((reg_train_times.size, res_feature_size+n+1, res_feature_size+n+1))
    Y_train = np.ascontiguousarray(u_arr_train[:, skip:-1])
    print('Y_train:')
    print(Y_train[-5,-5])
    reg_train_fracs = Y_train.shape[1]/reg_train_times
    print('Reg train fracs:')
    print(reg_train_fracs)
    if traintype in ['normal', 'normalres1', 'normalres2']:
        # Normal multi-noise training that sums all reservoir state outer products
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = X[:, skip:(res_d - 2)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            data_trstates += Y_train @ X_train.T
            states_trstates[0] += X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rmeanq','rmeanqres1','rmeanqres2']:
        # Training using the mean and the rescaled sum of the perturbations from the mean
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
                X_all = np.zeros((X_train.shape[0], X_train.shape[1], noise_realizations))
            X_train_mean += X_train/noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_mean.T
        states_trstates[0] = X_train_mean @ X_train_mean.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        prev_reg_train_time = 0
        for j, reg_train_time in enumerate(reg_train_times):
            for i in range(noise_realizations):
                Q_fit = X_all[:, prev_reg_train_time:reg_train_time, i] - \
                        X_train_mean[:,prev_reg_train_time:reg_train_time]
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += (Q_fit @ Q_fit.T)*reg_train_fracs[k]/noise_realizations
            prev_reg_train_time = reg_train_time
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rmean', 'rmeanres1', 'rmeanres2']:
        # Training using the mean only
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = X[:, skip:(res_d - 2)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
        data_trstates = Y_train @ X_train_mean.T
        states_trstates[0] = X_train_mean @ X_train_mean.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rqmean','rqmeanres1','rqmeanres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the mean
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip-1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates[0] = X_train_0 @ X_train_0.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        prev_reg_train_time = 0
        for j, reg_train_time in enumerate(reg_train_times):
            for i in range(noise_realizations):
                Q_fit = X_all[:,prev_reg_train_time:reg_train_time,i] - \
                        X_train_mean[:,prev_reg_train_time:reg_train_time]
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += (Q_fit @ Q_fit.T)*reg_train_fracs[k]/noise_realizations
            prev_reg_train_time = reg_train_time
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rq','rqres1','rqres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the noiseless reservoir
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip-1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,     leakage, noise[i], noisetype,noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates[0] = X_train_0 @ X_train_0.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        prev_reg_train_time = 0
        for j, reg_train_time in enumerate(reg_train_times):
            for i in range(noise_realizations):
                Q_fit = X_all[:,prev_reg_train_time:reg_train_time,i] - \
                        X_train_0[:,prev_reg_train_time:reg_train_time]
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += (Q_fit @ Q_fit.T)*reg_train_fracs[k]/noise_realizations
            prev_reg_train_time = reg_train_time
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rplusq', 'rplusqres1', 'rplusqres2']:
        # Training using the sum of the outer products of the sum of the noiseless reservoir and
        # the perturbation from the mean
        X_0, u_arr_train_noise = getXwrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip-1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
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
        states_trstates[0] = X_fit @ X_fit.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
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
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                noise_scaling, i, 'normal')
            X = X[:, skip:(res_d - 2)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip-1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
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
        states_trstates[0] += X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype == 'gradient':
        # Training using only the input jacobian (linearized 1-step noise)
        X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        #gradient_reg = np.zeros((res_feature_size+n+1, res_feature_size+n+1))
        #with objmode(Win = 'double[:,:]'):
        #    Win = csc_matrix((Win_data, Win_indices, Win_indptr), shape = (Win_shape[0], Win_shape[1])).toarray()
        Win = np.zeros((rsvr_size, n+1))
        for i in range(X.shape[1]):
            D_n_base = matrix_diag_mult(D[:, i], Win[:, 1:])
            if squarenodes:
                D_n = np.concatenate(
                    (D_n_base, matrix_diag_mult(2*X[:,i], D_n_base), np.identity(n)), axis=0)
            else:
                D_n = np.concatenate(
                     (D_n_base, np.identity(n)), axis=0)
            gradient_reg[0,1:,1:] += D_n @ D_n.T
        """
        print('Gradient reg:')
        print(gradient_reg[:5,:5])
        """
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype == 'gradient12':
        # Training using the input and reservoir jacobian (linearized 2-step noise)
        X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
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
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        #gradient_reg = np.zeros((res_feature_size+n+1, res_feature_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n, squarenodes)
        #with objmode(Win = 'double[:,:]'):
        #     Win = csc_matrix((Win_data, Win_indices, Win_indptr), shape = (Win_shape[0], Win_shape[1])).toarray()
        Win = np.zeros((rsvr_size, n+1))
        if squarenodes:
            leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                                np.identity(rsvr_size), np.zeros((rsvr_size, n+rsvr_size))), axis=1)
        else:
            leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                                 np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis=1)
        for i in range(1, X.shape[1]):
            if i > 0:
                D_n2_base = matrix_diag_mult(D[:, i-1], Win[:, 1:])
                if squarenodes:
                    D_n2 = np.concatenate((np.zeros((1, n)), D_n2_base, matrix_diag_mult(
                        2*X[:,i-1], D_n2_base), np.identity(n)), axis=0)
                else:
                    D_n2 = np.concatenate((np.zeros((1, n)), D_n2_base, np.identity(n)), axis=0)
                E_n_base = matrix_dot_left_T(W_mat_data, W_mat_indices,\
                    W_mat_indptr, W_mat_shape, np.diag(D[:, i])) + leakage_mat
                if squarenodes:
                    E_n = np.concatenate((np.zeros((1, rsvr_size+n+1)),\
                        E_n_base, matrix_diag_mult(2*X[:,i], E_n_base),\
                        np.zeros((n, rsvr_size+n+1))), axis=0)
                else:
                    E_n = np.concatenate((np.zeros((1, rsvr_size+n+1)),\
                        E_n_base, np.zeros((n, rsvr_size+n+1))), axis=0)
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
            D_n_base = matrix_diag_mult(D[:, i], Win[:, 1:])
            if squarenodes:
                D_n = np.concatenate(
                        (D_n_base, matrix_diag_mult(2*X[:,i], D_n_base), np.identity(n)), axis=0)
            else:
                D_n = np.concatenate((D_n_base, np.identity(n)), axis=0)
            """
            if i == 1:
                print('D_n:')
                print(D_n[:5,:5])
            if i == 2:
                print('Reg components @ 2,1')
                print(D_n[:5,:5])
            """
            gradient_reg[0,1:, 1:] += D_n @ D_n.T
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
        states_trstates[0] = X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype == 'gradient2':
        # Linearized 2-step noise with no 1-step noise
        X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0)
        #gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        #with objmode(Win = 'double[:,:]'):
        #     Win = csc_matrix((Win_data, Win_indices, Win_indptr), shape = (Win_shape[0], Win_shape[1])).toarray()
        Win = np.zeros((rsvr_size, n+1))
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
        states_trstates[0] = X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif 'gradientk' in traintype:
        # Linearized k-step noise
        k = str_to_int(traintype.replace('gradientk', ''))
        sparse_cutoff = 0.89
        break_flag = False
        #print(k)
        X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        gradient_reg_base = np.zeros((res_feature_size+n+1, res_feature_size+n+1))
        #D_n = np.zeros((res_feature_size+n+1, n, k))
        #E_n = np.zeros((res_feature_size+n+1, res_feature_size+n+1, k-1))
        #reg_components = np.zeros((res_feature_size+n+1, n, k))
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
            get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
        D_n_datas = List()
        D_n_indices = List()
        D_n_indptrs = List()
        D_n_shape = np.array([res_feature_size+n+1, n])
        E_n_datas = List()
        E_n_indices = List()
        E_n_indptrs = List()
        E_n_shape = np.array([res_feature_size+n+1, res_feature_size+n+1])
        #reg_comp_datas, reg_comp_indices, reg_comp_indptrs, reg_comp_shape = \
        #    [np.zeros(1, dtype = np.double)]*k,\
        #    [np.zeros(1, dtype = np.int32)]*k,\
        #    [np.zeros(1, dtype = np.int32)]*k,\
        reg_comp_datas   = List()
        reg_comp_indices = List()
        reg_comp_indptrs = List()
        reg_comp_shape   = np.array([res_feature_size+n+1, n])
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n, squarenodes)
        leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage, squarenodes)
        #if squarenodes:
        #    leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
        #                        np.identity(rsvr_size), np.zeros((rsvr_size, n+rsvr_size))), axis=1)
        #else:
        #    leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
        #                         np.identity(rsvr_size), np.zeros((rsvr_size, n))),   axis=1)
        reg_sum_avg_runtime = 0.
        E_n_avg_runtime = 0.
        reg_mult_avg_runtime = 0.
        D_n_avg_runtime = 0.
        #with objmode(tic = 'double'):
        #    tic = time.perf_counter()

        for i in range(k):
            D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:,i], X[:,i], Win_nobias_data, Win_nobias_indices,\
                Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size, res_feature_size, n, squarenodes)
            D_n_datas.append(np.ascontiguousarray(D_n_data))
            D_n_indices.append(np.ascontiguousarray(D_n_idx))
            D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
        for i in range(1, k):
            E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:,i], X[:,i], E_n_shape, rsvr_size, W_mat_data,\
                W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data, leakage_indices, \
                leakage_indptr, leakage_shape, squarenodes)
            E_n_datas.append(np.ascontiguousarray(E_n_data))
            E_n_indices.append(np.ascontiguousarray(E_n_idx))
            E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))
        #reg_comp_datas[k-1], reg_comp_indices[k-1], reg_comp_indptrs[k-1] =\
        #    np.copy(D_n_datas[-1]), np.copy(D_n_indices[-1]), np.copy(D_n_indptrs[-1])
        #reg_components[:, :, k-1] = D_n[:, :, -1]

        for i in range(k-1):
            #reg_components[:, :, i] = D_n[:, :, i]
            reg_comp_data, reg_comp_idx, reg_comp_indptr =\
                np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
            for j in range(i, k-1):
                #reg_components[:, :, i] = matrix_sparse_mult(
                #    E_n[:, :, j], reg_components[:, :, i])
                reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(\
                    E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,\
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
            reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
            reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
            reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
        reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
        reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
        reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
        sparsity = np.array([reg_comp_datas[j].size/(reg_comp_shape[0]*reg_comp_shape[1]) for j in range(k)])

        for i in range(k, X.shape[1]):
            #with objmode(itr_tic = 'double'):
            #    itr_tic = time.perf_counter()
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    #gradient_reg += reg_components[:, :, j] @ reg_components[:, :, j].T
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(\
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],\
                        reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = i+1 == reg_train_times
            if np.any(assign_grad_reg):
                gradient_reg[i+1 == reg_train_times] = gradient_reg_base*reg_train_fracs[i+1 == reg_train_times]
            if assign_grad_reg[-1]:
                break_flag = True
                break
            #print(gradient_reg[:5,:5])
            """
            with objmode(reg_sum_toc = 'double'):
                reg_sum_toc = time.perf_counter()
            if i > k:
                reg_sum_avg_runtime = reg_sum_avg_runtime*(i-k)/(i-k+1)+(reg_sum_toc-itr_tic)/(i-k+1)
            else:
                reg_sum_avg_runtime = reg_sum_toc-itr_tic
            """
            """
            if i % 200 == 0:
                with objmode():
                    runtime = time.perf_counter() - tic
                    print('Runtime at i = %d: %0.2f sec.' % (i, runtime))
                    print('Reg sum avg. runtime: %e' % reg_sum_avg_runtime)
                    print('E_n calc. avg. runtime: %e' % E_n_avg_runtime)
                    print('Reg mult avg. runtime: %e' % reg_mult_avg_runtime)
                    print('D_n calc. avg. runtime: %e' % D_n_avg_runtime)
            """
            #with objmode(E_n_tic = 'double'):
            #    E_n_tic = time.perf_counter()
            #E_n_datas[k-2], E_n_indices[k-2], E_n_indptrs[k-2] = get_E_n(D[:,i], X[:,i], \
            #    E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape,\
            #    leakage_data, leakage_indices, leakage_indptr, leakage_shape,\
            #    squarenodes)
            E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:,i], X[:,i], \
                E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape,\
                leakage_data, leakage_indices, leakage_indptr, leakage_shape,\
                squarenodes)
            E_n_datas[k-2] = np.ascontiguousarray(E_n_data)
            E_n_indices[k-2] = np.ascontiguousarray(E_n_idx)
            E_n_indptrs[k-2] = np.ascontiguousarray(E_n_indptr)
            #E_n_base = matrix_diag_sparse_mult_add(
            #    D[:, i], W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape,      leakage_mat)
            #if squarenodes:
            #    E_n[1+rsvr_size:1+res_feature_size,:,k-2] = matrix_diag_mult(2*X[:,  i], E_n_base)
            #E_n[1:rsvr_size+1, :, k-2] = E_n_base
            """
            with objmode(E_n_toc = 'double'):
                E_n_toc = time.perf_counter()
            if i > k:
                E_n_avg_runtime = E_n_avg_runtime*(i-k)/(i-k+1)+(E_n_toc-E_n_tic)/(i-k+1)
            else:
                E_n_avg_runtime = E_n_toc-E_n_tic
            with objmode(reg_mult_tic = 'double'):
                reg_mult_tic = time.perf_counter()
            """

            #reg_comp_datas_new, reg_comp_indices_new, reg_comp_indptrs_new = \
            #    [np.zeros(1, dtype = np.double) for x in range(0)],\
            #    [np.zeros(1, dtype = np.int32) for x in range(0)],\
            #    [np.zeros(1, dtype = np.int32) for x in range(0)]
            for j in range(k-1):
                if sparsity[j+1] < sparse_cutoff:
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(\
                        E_n_datas[k-2], E_n_indices[k-2], E_n_indptrs[k-2], E_n_shape,\
                        reg_comp_datas[j+1], reg_comp_indices[j+1], reg_comp_indptrs[j+1], reg_comp_shape)
                else:
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(\
                        E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,\
                        reg_comp_datas[j+1], reg_comp_indices[j+1], reg_comp_indptrs[j+1], reg_comp_shape)
                reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                    np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx),\
                    np.ascontiguousarray(reg_comp_indptr)
                #reg_comp_datas_new.append(reg_comp_data)
                #reg_comp_indices_new.append(reg_comp_idx)
                #reg_comp_indptrs_new.append(reg_comp_indptr)
                    #reg_components[:, :, j] = matrix_sparse_sparse_mult(
                    #    E_n[:, :, k-2], reg_components[:, :, j+1])
            """
            with objmode(reg_mult_toc = 'double'):
                reg_mult_toc = time.perf_counter()
            if i > k:
                reg_mult_avg_runtime = reg_mult_avg_runtime*(i-k)/(i-k+1)+(reg_mult_toc-reg_mult_tic)/(i-k+1)
            else:
                reg_mult_avg_runtime = reg_mult_toc-reg_mult_tic
            with objmode(D_n_tic = 'double'):
                D_n_tic = time.perf_counter()
            """
            #D_n_base = matrix_diag_mult(D[:, i], Win[:, 1:])
            #if squarenodes:
            #    reg_components[1+rsvr_size:1+res_feature_size,:,k-1] = matrix_diag_mult(2*X[:,i], D_n_base)
            #reg_components[1:rsvr_size+1, :, k - 1] = D_n_base
            #reg_components[1+res_feature_size:, :, k-1] = np.identity(n)
            reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:,i], X[:,i], \
                Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape, D_n_shape, \
                rsvr_size, res_feature_size, n, squarenodes)
            reg_comp_datas[k-1], reg_comp_indices[k-1], reg_comp_indptrs[k-1] = \
                np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx),\
                np.ascontiguousarray(reg_comp_indptr)
            #reg_comp_datas_new.append(reg_comp_data)
            #reg_comp_indices_new.append(reg_comp_idx)
            #reg_comp_indptrs_new.append(reg_comp_indptr)

            #reg_comp_datas, reg_comp_indices, reg_comp_indptrs = reg_comp_datas_new, reg_comp_indices_new, reg_comp_indptrs_new
            """
            with objmode(D_n_toc = 'double'):
                D_n_toc = time.perf_counter()
            if i > k:
                D_n_avg_runtime = D_n_avg_runtime*(i-k)/(i-k+1)+(D_n_toc-D_n_tic)/(i-k+1)
            else:
                D_n_avg_runtime = D_n_toc-D_n_tic
            """
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
        if not break_flag:
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    #gradient_reg += reg_components[:, :, j] @ reg_components[:, :, j].T
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(\
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],\
                        reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = i+1 == reg_train_times
            if np.any(assign_grad_reg):
                gradient_reg[i+1 == reg_train_times] = gradient_reg_base*reg_train_fracs[i+1 == reg_train_times]
        """
        print('Gradient reg:')
        print(gradient_reg[:5,:5])
        """
        """
        with objmode(toc = 'double'):
            toc = time.perf_counter()
        print('Gradient reg. compute time in sec.')
        print(toc - tic)
        """
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.ascontiguousarray(states_trstates[0])
        print('Target matrix:')
        print(data_trstates[:5,:5])
        print('Information matrix:')
        print(states_trstates[0,:5,:5])
        print('Gradient regularization:')
        print(gradient_reg[0,:5,:5])
        print('X_train:')
        print(X_train[:5,:5])
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype == 'sylvester':
        # Sylvester regularization w/o derivative
        X, p = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
        X_train = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0))
        data_trstates = Y_train @ X_train.T
        data_trstates[:, 1:rsvr_size+1] += noise_scaling**2/noise_realizations * \
            matrix_dot_left_T(W_mat_data, W_mat_indices,
                              W_mat_indptr, W_mat_shape, Win[:, 1:])
        states_trstates[0] = X_train @ X_train.T
        left_mat_base = -noise_scaling**2/noise_realizations * \
            (Win[:, 1:].T @ Win[:, 1:])
        left_mat = left_mat_base.reshape(1, left_mat_base.shape[0], left_mat_base.shape[1])
        return data_trstates, states_trstates, Y_train, X_train, left_mat
    elif traintype == 'sylvester_wD':
        # Sylvester regularization with derivative
        X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                           W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d-(skip+1))), X, u_arr_train[:, skip-1:-2]), axis=0)
        Dmean = mean_numba_axis1(D)
        temp_mat = np.diag(Dmean) @ Win[:, 1:]
        target_correction = matrix_dot_left_T(
            W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, temp_mat)
        left_mat_base = temp_mat.T @ temp_mat
        target_correction = noise_scaling**2/noise_realizations * target_correction
        left_mat_base = -noise_scaling**2/noise_realizations * left_mat_base
        left_mat = left_mat_base.reshape(1, left_mat_base.shape[0], left_mat_base.shape[1])
        data_trstates = Y_train @ X_train.T
        data_trstates[:, 1:rsvr_size+1] += target_correction
        states_trstates[0] = X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, left_mat
    else:
        # Noiseless training
        data_trstates = np.zeros((n, rsvr_size+1+n), dtype=np.float64)
        states_trstates[0] = np.zeros(
            (n+rsvr_size+1, n+rsvr_size+1), dtype=np.float64)
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg


@jit(nopython=True, fastmath=True)
def getD(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise, skip, noisetype='none',
         noise_scaling=0, noise_realizations=1, traintype='normal', squarenodes = False):
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    Y_train = u_arr_train[:, skip+1:]
    X, D = getXwrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                       W_indptr, W_shape, leakage, noise, 'none', 0, 0, 'gradient', squarenodes)
    return [D[:, skip:(res_d - 2)]]

# CONCATENATE ALL THE DATA BEFORE RUNNING REGRESSION


def predict(res, u0,  steps=1000, squarenodes = False):
    # Wrapper for the prediction function
    Y = predictwrapped(res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices,
                       res.W_indptr, res.W_shape, res.Wout, res.leakage, u0, steps, squarenodes)
    return Y


@jit(nopython=True, fastmath=True)
def predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, Wout, leakage, u0, steps, squarenodes = False):
    # Numba compatible prediction function
    Y = np.empty((Win_shape[1]-1, steps + 1))
    X = np.empty((res_X.shape[0], steps + 1))

    Y[:, 0] = u0
    X[:, 0] = res_X[:, -2]

    for i in range(0, steps):
        X[:, i+1] = (1-leakage)*X[:, i] + leakage*np.tanh(mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1.,
                     Y[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, X[:, i]))
        Y[:, i+1] = Wout @ get_squared_vec(np.concatenate((np.array([1.]), X[:, i+1], Y[:, i])),\
                X.shape[0], squarenodes)

    return Y

def get_test_data(test_stream, tau, num_tests, rkTime, split, system='lorenz'):
    # Function for obtaining test data sets used to validate reservoir performance
    # Uses an array of random number generators
    if system == 'lorenz':
        ic = test_stream[0].random(3)*2-1
        u0 = np.zeros(64)
    elif system in ['KS', 'KS_d2175']:
        ic = np.zeros(3)
        u0 = (test_stream[0].random(64)*2-1)*0.6
        u0 = u0 - np.mean(u0)
    transient = 2000
    u_arr_train_nonoise, u_arr_test, p, params = RungeKuttawrapped(x0=ic[0],
         y0=ic[1], z0=30*ic[2], tau=tau, T=rkTime+transient, ttsplit=split+transient, u0=u0, system=system)
    u_arr_train_nonoise = u_arr_train_nonoise[:,transient:]
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
        elif system in ['KS', 'KS_d2175']:
            ic = np.zeros(3)
            u0 = (test_stream[i].random(64)*2-1)*0.6
            u0 = u0 - np.mean(u0)
        u_arr_train_nonoise, rktest_u_arr_test[:, :, i], p, params = RungeKuttawrapped(x0=ic[0],
             y0=ic[1], z0=30*ic[2], T=rkTime+transient, ttsplit=split+transient, u0=u0, system=system, params=params)
        rktest_u_arr_train_nonoise[:,:,i] = u_arr_train_nonoise[:,transient:]
        """
        print('Test data %d' % i)
        print(rktest_u_arr_test[-3:,-3:,i])
        """

    return rktest_u_arr_train_nonoise, rktest_u_arr_test, params


def test(res, Wout_itr, squarenodes, noise_in, rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max, num_tests=100, rkTime=1000, split=3000, showMapError=False, showTrajectories=False, showHist=False, system='lorenz', params=np.array([[], []], dtype=np.complex128), pmap=False, max_valid_time = 500, savepred = False, save_time_rms = False):
    # Wrapper function for the numba compatible test function.

    # tic = time.perf_counter()
    stable_count, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist,           pmap_max, pmap_max_wass_dist = testwrapped(
        res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.Wout[Wout_itr], res.leakage,             rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max, num_tests, rkTime, split, noise_in,showMapError, showTrajectories, showHist, system, params=params, pmap=pmap, max_valid_time = max_valid_time, squarenodes = squarenodes, savepred = savepred, save_time_rms = save_time_rms)
    # toc = time.perf_counter()
    # runtime = toc - tic
    # print("Test " + str(i) + " valid time: " + str(j))

    return stable_count/num_tests, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist, pmap_max, pmap_max_wass_dist


@jit(nopython=True, fastmath=True)
def testwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, Wout, leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test,   true_pmap_max, num_tests, rkTime, split, noise_in, showMapError=True,   showTrajectories=True, showHist=True, system='lorenz', tau=0.1, params=np.array([[], []], dtype=np.complex128), pmap=False, max_valid_time = 500, squarenodes = False, savepred = False, save_time_rms = False):
    # Numba compatable function for testing trained reservoir performance against true system time series
    stable_count = 0
#SBATCH --output=log_files/{{JOB_NAME}}.log
    num_vt_tests = ((rkTime-split)) // max_valid_time
    valid_time = np.zeros((num_tests, num_vt_tests))
    max_rms = np.zeros(num_tests)
    mean_rms = np.zeros(num_tests)
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    wass_dist = np.zeros(num_tests)
    pmap_max = []
    pmap_max_wass_dist = np.zeros(num_tests)
    if savepred:
        preds = np.zeros((num_tests, rktest_u_arr_test.shape[0], (rkTime-split)+1))
    else:
        preds = np.empty((1,1,1), dtype = np.double)
    #mean_all = np.zeros((num_tests, (rkTime-split)+1))
    #variances_all = np.zeros((num_tests, (rkTime-split)+1))
    if save_time_rms:
        rms = np.zeros((num_tests, (rkTime-split)))
    else:
        rms = np.empty((1,1), dtype = np.double)

    # print(num_tests)
    for i in range(num_tests):
        with objmode(test_tic='double'):
            test_tic = time.perf_counter()
        res_X = (np.zeros((res_X.shape[0], split+2))*2 - 1)
        # print('Win')
        # print(Win[:3,:3])
        # print('A')
        # print(W[:3,:3])
        # print('Wout')
        # print(Wout[:3,:3])

        # sets res.X
        res_X, p = getXwrapped(np.ascontiguousarray(
            rktest_u_arr_train_nonoise[:, :, i]), res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise_in)
        pred_full = predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
                              Wout, leakage, u0=rktest_u_arr_test[:, 0, i], steps=(rkTime-split), squarenodes = squarenodes)
        """
        with objmode():
            print('Test %d pred:' % i)
        print(pred[-3:,-3:])
        """
        if savepred:
            preds[i] = pred_full
        error = np.zeros(max_valid_time)
        if pmap:
            if system == 'lorenz':
                calc_pred = np.stack((pred_full[0]*7.929788629895004,
                     pred_full[1]*8.9932616136662, pred_full[2]*8.575917849311919+23.596294463016896))
                # wass_dist[i] = wasserstein_distance_empirical(calc_pred.flatten(), true_trajectory.flatten())
                pred_pmap_max = poincare_max(calc_pred, np.arange(pred_full.shape[0]))
            elif system == 'KS':
                # wass_dist[i] = wasserstein_distance_empirical(pred.flatten()*1.1876770355823614, true_trajectory.flatten())
                pred_pmap_max = poincare_max(pred_full*1.1876770355823614, np.arange(pred_full.shape[0]))
            elif system == 'KS_d2175':
                pred_pmap_max = poincare_max(pred_full*1.2146066380280796, np.arange(pred_full.shape[0]))

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
        pred = pred_full[:,:max_valid_time]
        for k in range(num_vt_tests):
            for j in range(1, pred.shape[1]):
                error[j] = np.sqrt(
                    np.mean((pred[:, j]-rktest_u_arr_test[:, k*max_valid_time+j, i])**2.0))

                if error[j] < vt_cutoff and check_vt:
                    valid_time[i,k] = j
                else:
                    #if check_vt:
                    """
                    print(j)
                    print('Prediction at VT overflow:')
                    print(pred[:5,j])
                    print('Truth at VT overflow:')
                    print(rktest_u_arr_test[:5,k*max_valid_time+j,i])
                    """
                    check_vt = False
            print('Valid Time')
            print(valid_time[i,k])
            res_X = np.zeros((res_X.shape[0], max_valid_time+2))
            res_X, p = getXwrapped(np.ascontiguousarray(
                rktest_u_arr_test[:, k*max_valid_time:(k+1)*max_valid_time+1, i]), res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise_in)
            """
            print('Res value after synchronization:')
            print(res_X[:5,-2])
            """
            if k < (num_vt_tests - 1):
                pred = predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
                    Wout, leakage, u0=rktest_u_arr_test[:, (k+1)*max_valid_time, i], steps=max_valid_time-1, squarenodes = squarenodes)
                check_vt = True
        if array_compute:
            if system == 'lorenz':
                rkmap_u_arr_train = RungeKuttawrapped_pred(u0_array=np.stack((pred_full[0]*7.929788629895004,
                    pred_full[1]*8.9932616136662, pred_full[2]*8.575917849311919+23.596294463016896)),
                    h=0.01, system=system, params=params, tau=tau, ttsplit=pred_full.shape[1])[0]
            elif system == 'KS':
                u0 = pred_full*1.1876770355823614
                rkmap_u_arr_train = RungeKuttawrapped_pred(
                    u0_array=u0, h=tau, T=1, system=system, params=params, ttsplit=pred_full.shape[1])[0]
            elif system == 'KS_d2175':
                u0 = pred_full*1.2146066380280796
                rkmap_u_arr_train = RungeKuttawrapped_pred(
                    u0_array=u0, h=tau, T=1, system=system, params=params, ttsplit=pred_full.shape[1])[0]
            # print(rkmap_u_arr_train[0,:10])
            x2y2z2 = sum_numba_axis0(
                (pred_full[:, 1:]-rkmap_u_arr_train[:, :-1])**2.0)
        else:
            x2y2z2 = np.zeros(pred_full[0].size-1)
            for j in range(1, pred_full[0].size):

                if system == 'lorenz':
                    rkmap_u_arr_train = RungeKuttawrapped(pred_full[0][j-1]*7.929788629895004, pred_full[1][j-1]*8.9932616136662, pred_full[2]
                                                          [j-1]*8.575917849311919+23.596294463016896, h=0.01, T=1, tau=tau, system=system, params=params)[0]
                elif system == 'KS':
                    u0 = pred_full[:, j-1]*(1.1876770355823614)
                    rkmap_u_arr_train = RungeKuttawrapped(
                        0, 0, 0, h=tau, T=1, u0=u0, system=system, params=params)[0]
                elif system == 'KS_d2175':
                    u0 = pred_full[:, j-1]*(1.2146066380280796)
                    rkmap_u_arr_train = RungeKuttawrapped(
                        0, 0, 0, h=tau, T=1, u0=u0, system=system, params=params)[0]
                # if j <= 10:
                # print(rkmap_u_arr_train[0,1])

                x2y2z2[j-1] = np.sum((pred_full[:, j]-rkmap_u_arr_train[:, 1])**2)
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
        rms_test = np.sqrt(x2y2z2/pred_full.shape[0])
        if save_time_rms:
            rms[i] = rms_test
        max_rms[i] = np.max(rms_test)
        mean_rms[i] = np.mean(rms_test)
        if system == 'lorenz':
            means[i] = np.mean(pred_full[0])
            variances[i]     = np.var(pred_full[0])
            #mean_all[i]      = sum_numba_axis0(pred)/pred.shape[0]
            #variances_all[i] = numba_var_axis0(pred)
        elif system in ['KS', 'KS_d2175']:
            means[i] = np.mean(pred_full.flatten())
            variances[i] = np.var(pred_full.flatten())
            #mean_all[i] = sum_numba_axis0(pred)/pred.shape[0]
            #variances_all[i] = numba_var_axis0(pred)

        # print('Map error: ', mean_rms[i])
        # print('Variance: ', variances[i])
        # print('True Variance: ', np.var(rktest_u_arr_test))
        if mean_rms[i] < 5e-3 and 0.9 < variances[i] and variances[i] < 1.1:
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

    print("Avg. max sum square: " + str(np.mean(max_rms)))
    print("Avg. mean sum square: " + str(np.mean(mean_rms)))
    print("Avg. of x dim: " + str(np.mean(means)))
    print("Var. of x dim: " + str(np.mean(variances)))
    print()
    """
    print('Mean rms shape:')
    print(mean_rms.shape)
    #return stable_count, mean_rms, max_rms, variances, valid_time, mean_all, variances_all, preds, wass_dist, pmap_max, pmap_max_wass_dist
    return stable_count, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist,    pmap_max, pmap_max_wass_dist

def generate_res(res_gen, res_itr, squarenodes, rk, reg_train_times, res_size, rho, sigma, leakage, win_type, bias_type, noise_stream, noisetype='none', noise_scaling=0, noise_realizations=1, traintype='normal', skip=500):
    # Function for generating a reservoir and obtaining matrices used for training the reservoir
    reservoir = Reservoir(rk, res_gen, res_itr, rk.u_arr_train.shape[0], rsvr_size=res_size,
                spectral_radius=rho, input_weight=sigma, leakage=leakage, win_type=win_type, bias_type=bias_type)
    # print('Train Data shape: (%d, %d)' % (rk.u_arr_train.shape[0], rk.u_arr_train.shape[1]))
    # print(rk.u_arr_train[-3:,-3:])
    data_shape = rk.u_arr_train.shape
    res_shape = reservoir.X.shape
    noise_in = gen_noise_driver(data_shape, res_shape, traintype,
                                noisetype, noise_scaling, noise_stream, noise_realizations)
    # print(noise_in.shape)
    get_states(reservoir, squarenodes, rk, reg_train_times, noise_in, noisetype, noise_scaling,
               noise_realizations, traintype, skip)
    return reservoir, noise_in


def optim_func(res, squarenodes, noise_in, noise, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha,   true_pmap_max, rkTime=400, split=2000, traintype='normal', system='lorenz', params=np.array([[], []], dtype=np.complex128), pmap = False, max_valid_time = 500, savepred = False, save_time_rms = False):
    # Function for training and testing the performance of a reservoir trained using a particular regularization parameter
    if squarenodes:
        res_feature_size = res.rsvr_size*2
    else:
        res_feature_size = res.rsvr_size
    idenmat = np.identity(
        res_feature_size+1+rktest_u_arr_train_nonoise.shape[0])*alpha
    print('Gradient reg shape')
    print(res.gradient_reg.shape)
    num_reg_train_times = res.gradient_reg.shape[0]
    res.Wout = np.zeros((num_reg_train_times, rktest_u_arr_train_nonoise.shape[0],\
            rktest_u_arr_train_nonoise.shape[0]+1+res_feature_size))
    results            = np.zeros(num_reg_train_times, dtype = object)
    mean_rms           = np.zeros(num_reg_train_times, dtype = object)
    max_rms            = np.zeros(num_reg_train_times, dtype = object)
    variances          = np.zeros(num_reg_train_times, dtype = object)
    valid_time         = np.zeros(num_reg_train_times, dtype = object)
    rms                = np.zeros(num_reg_train_times, dtype = object)
    preds              = np.zeros(num_reg_train_times, dtype = object)
    wass_dist          = np.zeros(num_reg_train_times, dtype = object)
    pmap_max           = np.zeros(num_reg_train_times, dtype = object)
    pmap_max_wass_dist = np.zeros(num_reg_train_times, dtype = object)
    for i in range(num_reg_train_times):
        if traintype not in ['sylvester', 'sylvester_wD']:
            # print('Noise mag: %e' % noise)
            # print('Gradient reg:')
            # print(res.gradient_reg[:5,:5])
            #print(res.states_trstates.shape)
            #print(res.gradient_reg.shape)
            #print(idenmat.shape)
            res.Wout[i] = np.transpose(solve(np.transpose(
                res.states_trstates[i] + noise**2.0*res.gradient_reg[i]+idenmat), np.transpose(res.data_trstates)))
            #res.Wout =  matlab_mrdivide(res.states_trstates + noise**2.0*res.gradient_reg + idenmat, res.data_trstates)
        else:
            res.Wout[i] = solve_sylvester(
                res.left_mat[i], res.states_trstates[i]+idenmat, res.data_trstates)

        train_rms = np.sqrt(np.mean((res.Wout[i] @ res.X_train - res.Y_train)**2.0, axis = 0))
        train_mean_rms = np.mean(train_rms)
        train_max_rms  = np.max(train_rms)
        results[i], mean_rms[i], max_rms[i], variances[i], valid_time[i],  rms[i], preds[i], \
            wass_dist[i], pmap_max[i], pmap_max_wass_dist[i] = test(res, i, squarenodes, noise_in, \
            rktest_u_arr_train_nonoise, rktest_u_arr_test,   true_pmap_max, num_tests=num_tests, rkTime=rkTime,\
            split=split, showMapError=True, showTrajectories=True, showHist=True, system=system, params=params,\
            pmap = pmap, max_valid_time = max_valid_time, savepred = savepred, save_time_rms = save_time_rms)

    return -1*results, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist, pmap_max, pmap_max_wass_dist, train_mean_rms, train_max_rms


def get_res_results(itr, res_gen, squarenodes, rk, reg_train_times, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, noise_stream, traintype,
    rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha_values, rkTime_test, split_test, system, tau, params, savepred, save_time_rms, debug_mode, train_seed,   true_pmap_max, pmap, max_valid_time):
    # Function for generating, training, and testing the performance of a reservoir given an input set of testing data time series,
    # a set of regularization values, and a set of noise magnitudes
    tic = time.perf_counter()
    print('Starting res %d' % itr)
    reservoir, noise_in = generate_res(res_gen, itr, squarenodes, rk, reg_train_times, res_size, rho, sigma, leakage,
                                       win_type, bias_type, noise_stream, noisetype, noise, noise_realizations, traintype)

    toc = time.perf_counter()
    print('Res states found for itr %d, runtime: %f sec.' % (itr, toc-tic))
    num_vt_tests = (rkTime_test - split_test) // max_valid_time

    #final_out = []
    #final_out.append((np.copy(stable_frac_0), np.copy(stable_frac), np.copy(mean_rms_0), np.copy(mean_rms),\
    #             np.copy(max_rms_0), np.copy(max_rms), np.copy(variances_0), np.copy(variances),  \
    #             np.copy(valid_time_0), np.copy(valid_time), \
    #             np.copy(rms_0), np.copy(rms), np.copy(preds), train_seed, noise, \
    #             itr, np.copy(wass_dist_0), np.copy(wass_dist), np.copy(pmap_max_wass_dist), np.copy(train_mean_rms),\
    #             np.copy(train_max_rms)))

    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise
    stable_frac_out        = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    mean_rms_out           = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    max_rms_out            = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    variances_out          = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    valid_time_out         = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    rms_out                = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    preds_out              = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    wass_dist_out          = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    pmap_max_wass_dist_out = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    stable_frac_out        = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    train_mean_rms_out     = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    train_max_rms_out      = np.zeros((noise_array.size, alpha_values.size, reg_train_times.size), dtype = object)
    print('Mean RMS out shape:')
    print(mean_rms_out.shape)
    for i, noise in enumerate(noise_array):
        """
        stable_frac = np.zeros((alpha_values.size))
        mean_rms = np.zeros((num_tests, alpha_values.size))
        max_rms = np.zeros((num_tests, alpha_values.size))
        variances = np.zeros((num_tests, alpha_values.size))
        valid_time = np.zeros((num_tests, num_vt_tests, alpha_values.size))
        wass_dist = np.zeros((num_tests, alpha_values.size))
        train_mean_rms = np.zeros((alpha_values.size))
        train_max_rms  = np.zeros((alpha_values.size))

        if save_time_rms:
            #mean_all = np.zeros((num_tests, (rkTime_test-split_test)+1, alpha_values.size-1))
            #variances_all = np.zeros((num_tests, (rkTime_test-split_test)+1, alpha_values.size-1))
            rms = np.zeros((num_tests, (rkTime_test-split_test), alpha_values.size))
        # pmap_max           = []
        pmap_max_wass_dist = np.zeros((num_tests, alpha_values.size))
        """
        noise_tic = time.perf_counter()
        min_optim_func = lambda alpha: optim_func(reservoir, squarenodes, noise_in[0], noise, \
                rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha,\
                true_pmap_max, rkTime_test, split_test, traintype, system, params, pmap,\
                max_valid_time, savepred, save_time_rms)
        func_vals = np.zeros(alpha_values.size)
        for j, alpha_value in enumerate(alpha_values):
            print('Regularization: ', alpha_values[j])
            if debug_mode:
                #out = min_optim_func(alpha_values[j])
                stable_frac_out[i,j], mean_rms_out[i,j], max_rms_out[i,j], variances_out[i,j],\
                        valid_time_out[i,j], rms_out[i,j], preds_out[i,j], wass_dist_out[i,j],\
                        tmp, pmap_max_wass_dist_out[i,j], train_mean_rms_out[i,j], \
                        train_max_rms_out[i,j] = min_optim_func(alpha_value)
                """
                train_mean_rms[j] = out[-2]
                train_max_rms[j]  = out[-1]
                if j == 0:
                    stable_frac_0 = out[0]
                    variances_0 = out[3]
                    mean_rms_0 = out[1]
                    max_rms_0 = out[2]
                    valid_time_0 = out[4]
                    if save_time_rms:
                        #mean_all_0 = out[5]
                        #variances_all_0 = out[6]
                        rms_0 = out[5]
                    wass_dist_0 = out[7]
                    # restart editing here
                else:
                    stable_frac[j-1] = out[0]
                    variances[:, j-1] = out[3]
                    mean_rms[:, j-1] = out[1]
                    max_rms[:,j-1] = out[2]
                    valid_time[:,:,j-1] = out[4]
                    if save_time_rms:
                        #mean_all[:,:,j-1] = out[5]
                        #variances_all[:,:,j-1] = out[6]
                        rms[:,:,j-1] = out[5]
                    wass_dist[:, j-1] = out[7]
                    # pmap_max.append(out[7])
                    pmap_max_wass_dist[:, j-1] = out[9]
                if savepred:
                    if j == 0:
                        preds = out[6]
                    elif j == 1:
                        preds = np.stack((preds, out[6]), axis=3)
                    elif j > 1:
                        preds = np.concatenate((preds, out[6].reshape(
                            out[6].shape[0], out[6].shape[1], out[6].shape[2], 1)), axis=3)
                """
            else:
                try:
                    stable_frac_out[i,j], mean_rms_out[i,j], max_rms_out[i,j], variances_out[i,j],\
                        valid_time_out[i,j], rms_out[i,j], preds_out[i,j], wass_dist_out[i,j],\
                        tmp, pmap_max_wass_dist_out[i,j], train_mean_rms_out[i,j], \
                        train_max_rms_out[i,j] = min_optim_func(alpha_value)
                    """
                    out = min_optim_func(alpha_values[j])
                    train_mean_rms[j] = out[-2]
                    train_max_rms[j]  = out[-1]
                    if j == 0:
                        stable_frac_0 = out[0]
                        variances_0 = out[3]
                        mean_rms_0 = out[1]
                        max_rms_0 = out[2]
                        valid_time_0 = out[4]
                        if save_time_rms:
                            #mean_all_0 = out[5]
                            #variances_all_0 = out[6]
                            rms_0 = out[5]
                        wass_dist_0 = out[7]
                    else:
                        stable_frac[j-1] = out[0]
                        variances[:, j-1] = out[3]
                        mean_rms[:, j-1] = out[1]
                        max_rms[:, j-1] = out[2]
                        valid_time[:,:,j-1] = out[4]
                        if save_time_rms:
                            #mean_all[:,:,j-1] = out[5]
                            #variances_all[:,:,j-1] = out[6]
                            rms[:,:,j-1] = out[5]
                        wass_dist[:, j-1] = out[7]
                        # pmap_max.append(out[7])
                        pmap_max_wass_dist[:, j-1] = out[9]
                    if savepred:
                        if j == 0:
                            preds = out[6]
                        elif j == 1:
                            preds = np.stack((preds, out[6]), axis=3)
                        elif j > 1:
                            preds = np.concatenate((preds, out[6].reshape(
                                out[6].shape[0], out[6].shape[1], out[6].shape[2], 1)), axis=3)
                    """
                except:
                    print('Training unsucessful for alpha:')
                    print(alpha_values[j])
                    """
                    train_mean_rms[j] = 0.
                    train_max_rms[j]  = 0.
                    if j == 0:
                        stable_frac_0 = 0.0
                        variances_0 = np.zeros(num_tests)
                        mean_rms_0 = np.zeros(num_tests)
                        max_rms_0 = np.zeros(num_tests)
                        valid_time_0 = np.zeros((num_tests, num_vt_tests))
                        if save_time_rms:
                            #mean_all_0  = np.zeros((num_tests, (rkTime-split)+1))
                            #variances_all_0  = np.zeros((num_tests, (rkTime-split)+1))
                            rms_0 = np.zeros((num_tests, (rkTime-split)))
                        wass_dist_0 = np.zeros(num_tests)
                    else:
                        stable_frac[j-1] = 0.0
                        variances[:, j-1] = np.zeros(num_tests)
                        mean_rms[:, j-1] = np.zeros(num_tests)
                        max_rms[:, j-1] = np.zeros(num_tests)
                        valid_time[:,:, j-1] = np.zeros((num_tests, num_vt_tests))
                        wass_dist[:, j-1] = np.zeros(num_tests)
                        if save_time_rms:
                            #mean_all[:,:,j-1]= np.zeros((num_tests, (rkTime-split)+1))
                            #variances_all[:,:,j-1]= np.zeros((num_tests, (rkTime-split)+1))
                            rms[:,:,j-1] = np.zeros((num_tests, (rkTime-split)))
                        # pmap_max.append(np.empty((2,2), dtype = np.float64))
                        pmap_max_wass_dist[:, j-1] = np.zeros(num_tests)
                    if savepred:
                        pred_out = np.zeros(
                            (num_tests, rktest_u_arr_test.shape[0], (rkTime_test-split_test)+1))
                        if j == 0:
                            preds = preds_out
                        elif j == 1:
                            preds = np.stack((preds, preds_out), axis=3)
                        elif j > 1:
                            preds = np.concatenate((preds, preds_out.reshape(
                                preds_out.shape[0], preds_out.shape[1], preds_out.shape[2], 1)), axis=3)
                    """
        """
        if not savepred:
            preds = np.empty((1, 1, 1, 1), dtype=np.complex128)
        """
        """
        if not save_time_rms:
            #mean_all_0 = np.empty((1,1), dtype=np.float64)
            #mean_all   = np.empty((1,1,1), dtype=np.float64)
            #variances_all_0 = np.empty((1,1), dtype=np.float64)
            #variances_all   = np.empty((1,1,1), dtype=np.float64)
            rms_0 = np.empty((1,1), dtype=np.float64)
            rms   = np.empty((1,1,1), dtype=np.float64)
        """

        """
        final_out.append((np.copy(stable_frac_0), np.copy(stable_frac), np.copy(mean_rms_0), np.copy(mean_rms),\
                np.copy(max_rms_0), np.copy(max_rms), np.copy(variances_0), np.copy(variances),  \
                np.copy(valid_time_0), np.copy(valid_time), \
                np.copy(rms_0), np.copy(rms), np.copy(preds), train_seed, noise, \
                itr, np.copy(wass_dist_0), np.copy(wass_dist), np.copy(pmap_max_wass_dist), np.copy(train_mean_rms),\
                np.copy(train_max_rms)))  # pmap_max, np.copy(pmap_max_wass_dist)))
        """
        #print(type(mean_rms_out[0,0,1]))
        #print('Mean rms shape:')
        #print(mean_rms_out[0,0,1].shape)
        noise_toc = time.perf_counter()
        print('Noise test time: %f sec.' % (noise_toc - noise_tic))
    toc = time.perf_counter()
    runtime = toc - tic
    print('Iteration runtime: %f sec.' % runtime)
    return stable_frac_out, mean_rms_out, max_rms_out, variances_out, valid_time_out, rms_out, preds_out, wass_dist_out,\
            pmap_max_wass_dist_out, train_mean_rms_out, train_max_rms_out, train_seed, noise_array, itr


def find_stability(noisetype, noise, traintype, train_seed, train_gen, res_itr, res_gen, squarenodes, test_stream, noise_stream, rho, sigma, leakage, win_type, bias_type, train_time, reg_train_times, res_size, res_per_test, noise_realizations, num_tests, alpha_values, system, tau, savepred, save_time_rms, debug_mode, root_folder, pmap, max_valid_time, raw_data_folder):

    # Main function for training and testing reservoir performance. This function first generates the training and testing data,
    # the passes to get_res_results to obtain th reservoir performance.

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
    elif system in ['KS', 'KS_d2175']:
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
    elif system in ['KS', 'KS_d2175']:
        u0 = 0.6*(train_gen.random(64)*2-1)
        u0 = u0 - np.mean(u0)
        rk = RungeKutta(0, 0, 0, tau=tau, T=train_time,
                        ttsplit=train_time, u0=u0, system=system, params=params)
    #print('Training data %d:' % train_seed)
    #print(rk.u_arr_train[-3:,-3:])
    #true_filename = root_folder + \
    #    '%s_tau%0.2f_true_trajectory.csv' % (system, tau)
    #true_trajectory = np.loadtxt(true_filename, delimiter=',')
    if pmap:
        true_pmap_max_filename = root_folder + \
            '%s_tau%0.2f_true_pmap_max.csv' % (system, tau)
        true_pmap_max = np.loadtxt(true_pmap_max_filename, delimiter=',')
        print('Snippet of true poincare map:')
        print(true_pmap_max[:5])
    else:
        true_pmap_max = np.zeros(100)

    stable_frac_out, mean_rms_out, max_rms_out, variances_out, valid_time_out, rms_out, pred_out, wass_dist_out,\
            pmap_max_wass_dist_out, train_mean_rms_out, train_max_rms_out, train_seed, noise_array, itr\
            = get_res_results(res_itr, res_gen, squarenodes, rk, reg_train_times, res_size, rho, sigma, \
            leakage, win_type, bias_type, noisetype, noise, noise_realizations, noise_stream, \
            traintype, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha_values, \
            rkTime_test, split_test, system, tau, params, savepred, save_time_rms, debug_mode, \
            train_seed, true_pmap_max, pmap, max_valid_time)
    print(type(stable_frac_out))
    print(stable_frac_out.shape)
    #print(type(stable_frac_out[0,:,1]))
    #print(stable_frac_out[0,:,1].shape)

    print(type(mean_rms_out))
    print(mean_rms_out.shape)
    #print(type(mean_rms_out[0,:,1]))
    #print(mean_rms_out[0,:,1].shape)

    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise


    for (i, noise_val), (j, reg_train_time) in product(enumerate(noise_array), enumerate(reg_train_times)):
        print((i,j))
        stable_frac = np.zeros(alpha_values.size)
        for k, array_elem in enumerate(stable_frac_out[i,:,j]):
            stable_frac[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'stable_frac_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            stable_frac, delimiter = ',')

        mean_rms = np.zeros((alpha_values.size, *mean_rms_out[i,0,j].shape))
        for k, array_elem in enumerate(mean_rms_out[i,:,j]):
            mean_rms[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'mean_rms_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            mean_rms, delimiter = ',')

        max_rms = np.zeros((alpha_values.size, *max_rms_out[i,0,j].shape))
        for k, array_elem in enumerate(max_rms_out[i,:,j]):
            max_rms[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            max_rms, delimiter = ',')

        variances = np.zeros((alpha_values.size, *variances_out[i,0,j].shape))
        for k, array_elem in enumerate(variances_out[i,:,j]):
            variances[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'variance_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            variances, delimiter = ',')

        valid_time = np.zeros((alpha_values.size, *valid_time_out[i,0,j].shape))
        for k, array_elem in enumerate(valid_time_out[i,:,j]):
            valid_time[k] = array_elem
        for k in range(num_tests):
            np.savetxt(os.path.join(raw_data_folder, 'valid_time_res%d_train%d_test%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, k, noise_val, reg_train_time)),\
                valid_time[:,k], delimiter = ',')

        train_mean_rms = np.zeros((alpha_values.size, *train_mean_rms_out[i,0,j].shape))
        for k, array_elem in enumerate(train_mean_rms_out[i,:,j]):
            train_mean_rms[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'train_mean_rms_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            train_mean_rms, delimiter = ',')

        train_max_rms = np.zeros((alpha_values.size, *train_max_rms_out[i,0,j].shape))
        for k, array_elem in enumerate(train_max_rms_out[i,:,j]):
            train_max_rms[k] = array_elem
        np.savetxt(os.path.join(raw_data_folder, 'train_max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
            train_max_rms, delimiter = ',')

        if pmap:
            pmap_max_wass_dist = np.zeros((alpha_values.size, *pmap_max_wass_dist_out[i,0,j].shape))
            for k, array_elem in enumerate(pmap_max_wass_dist_out[i,:,j]):
                pmap_max_wass_dist[k] = array_elem
            np.savetxt(os.path.join(raw_data_folder, 'pmap_max_wass_dist_res%d_train%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, noise_val, reg_train_time)),\
                pmap_max_wass_dist, delimiter = ',')
        if save_time_rms:
            rms = np.zeros((alpha_values.size, *rms_out[i,0,j].shape))
            for k, array_elem in enumerate(rms_out[i,:,j]):
                rms[k] = array_elem
            for k in range(num_tests):
                #np.savetxt(os.path.join(raw_data_folder, 'mean_all_res%d_train%d_test%d_noise%e.csv' % (res_itr, train_seed, j, noise_val)),\
                #    np.hstack((out[i][10][j].reshape(-1,1), out[i][11][j])), delimiter = ',')
                #np.savetxt(os.path.join(raw_data_folder, 'variances_all_res%d_train%d_test%d_noise%e.csv' % (res_itr, train_seed, j, noise_val)),\
                #    np.hstack((out[i][12][j].reshape(-1,1), out[i][13][j])), delimiter = ',')
                np.savetxt(os.path.join(raw_data_folder, 'rms_res%d_train%d_test%d_noise%e_regtrain%d.csv' % (res_itr, train_seed, k, noise_val, reg_train_time)),\
                        rms[:,k], delimiter = ',')
        if savepred:
            pred = np.zeros((alpha_values.size, *pred_out[i,0,j].shape))
            for k, array_elem in enumerate(pred_out[i,:,j]):
                pred[k] = array_elem
            for l, (k, reg) in product(range(num_tests), enumerate(alpha_values)):
                np.savetxt(os.path.join(raw_data_folder, 'pred_res%d_train%d_test%d_noise%e_regtrain%d_reg%e.csv' %\
                        (res_itr, train_seed, l, noise_val, reg_train_time, reg)), pred[k,l], delimiter = ',')

    # toc_global = time.perf_counter()
    # print('Total Runtime: %s sec.' % (toc_global - tic_global))

    return rkTime_test, split_test

# If we are running reservoir tests in parallel, compile the find_stability as a remote function.
# Otherwise, wrap it.

@ray.remote
def find_stability_remote(*args):
    return find_stability(*args)

def find_stability_serial(*args):
    return find_stability(*args)

def main(argv):

    # Main driver function for obtaining reservoir performance. This function processes input arguments and
    # creates random number generators for the different reservoirs, trainings, noise arrays, and tests.
    # It then calls find_stability in a loop, processes the output from find_stability, and saves the output to a folder.

    root_folder, top_folder, run_name, system, noisetype, traintype, savepred, save_time_rms, squarenodes, rho,\
        sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, \
        noise_streams_per_test, noise_values_array,alpha_values, res_per_test, num_trains, num_tests,\
        debug_mode, pmap, metric, return_all, ifray, machine,max_valid_time, tmp, tmp1, tmp2, tmp3\
        ,reg_train_fracs, discard_time = get_run_opts(argv)

    raw_data_folder = os.path.join(os.path.join(root_folder, top_folder), run_name + '_folder')
    if machine == 'skynet':
        if ifray:
            ray.init(num_cpus = num_cpus)
    elif machine == 'deepthought2':
        if ifray:
            ray.init(address=os.environ["ip_head"])


    #noise_values_array = np.logspace(-3, 0, num = 19, base = 10)[5:11]
    #noise_values_array = np.logspace(-4, 0, num = 13, base = 10)[6:8]
    # noise_values_array = np.array([np.logspace(-4, 0, num = 13, base = 10)[6]])
    # noise_values_array = np.array([0,1e-3,1e-2])
    #alpha_values = np.append(0., np.logspace(-7, -3, 9))
    #alpha_values = np.array([0., 1e-6, 1e-4])
    if traintype in ['normal','normalres1','normalres2','rmean','rmeanres1','rmeanres2',
            'rplusq','rplusqres1','rplusqres2']:
        alpha_values = alpha_values*noise_realizations
    # alpha_values = np.array([0,1e-6,1e-4])
    reg_train_times = np.array([int((train_time-discard_time)*i) for i in \
            np.arange(1./reg_train_fracs, 1+1./reg_train_fracs, 1./reg_train_fracs)])
    np.savetxt(os.path.join(raw_data_folder, 'reg_train_times.csv'),   reg_train_times,    delimiter = ',')
    np.savetxt(os.path.join(raw_data_folder, 'test_noise_values.csv'), noise_values_array, delimiter = ',')
    np.savetxt(os.path.join(raw_data_folder, 'test_alpha_values.csv'), alpha_values      , delimiter = ',')

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
                tr[i], train_streams[i], rt[i], res_streams[i], squarenodes, test_streams[i], noise_streams, rho, sigma, \
                leakage, win_type, bias_type, train_time, reg_train_times,\
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, save_time_rms, debug_mode, \
                root_folder, pmap, max_valid_time, raw_data_folder) for i in range(tr.size)])
        else:
            out_base  = [find_stability_serial(noisetype, noise_values_array, traintype, \
                tr[i], train_streams[i], rt[i], res_streams[i], squarenodes, test_streams[i], noise_streams, rho, sigma, \
                leakage, win_type, bias_type, train_time, reg_train_times, \
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, save_time_rms, debug_mode,\
                root_folder, pmap, max_valid_time, raw_data_folder) for i in range(tr.size)]

        # print('Ray out len: %d' % len(out_base))
        # print('Out elem len: %d' % len(out_base[0]))

        """

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
        """
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
                tnr[i], train_streams[i], rtn[i], res_streams[i], squarenodes, test_streams[i], noise_streams[i], \
                rho, sigma, leakage, win_type, bias_type, train_time, reg_train_times,\
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, save_time_rms, debug_mode, root_folder, pmap, max_valid_time, raw_data_folder) for i in range(tnr.size)])
        else:
            out_base  = [find_stability_serial(noisetype, ntr[i], traintype,\
                tnr[i], train_streams[i], rtn[i], res_streams[i], squarenodes, test_streams[i], noise_streams[i], \
                rho, sigma, leakage, win_type, bias_type, train_time, reg_train_times,\
                res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
                system, tau, savepred, save_time_rms, debug_mode, root_folder, pmap, max_valid_time, raw_data_folder) for i in range(tnr.size)]
        """
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
        """
    rkTime, split = out_base[0]
    np.savetxt(os.path.join(raw_data_folder, 'test_time_split.csv'), np.array([rkTime, split]), delimiter = ',')

    # print(len(results[0]))
    ray.shutdown()
    toc = time.perf_counter()
    runtime = toc - tic
    print('Runtime over all cores: %f sec.' %(runtime))
    print('Ray finished.')
    print('Results Saved')


if __name__ == "__main__":
    main(sys.argv[1:])

