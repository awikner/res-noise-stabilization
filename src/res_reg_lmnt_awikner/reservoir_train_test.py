from numba.core.errors import NumbaPerformanceWarning
import warnings
import ray
import sys
import os

from scipy.linalg import solve, solve_sylvester
from scipy.sparse.linalg import eigsh
from numba import njit
from numba.typed import List
import math
import time

from res_reg_lmnt_awikner.lorenzrungekutta_numba import lorenzrungekutta, lorenzrungekutta_pred
from res_reg_lmnt_awikner.ks_etdrk4 import kursiv_predict, kursiv_predict_pred
from res_reg_lmnt_awikner.csc_mult import *
from res_reg_lmnt_awikner.helpers import get_windows_path, poincare_max
from res_reg_lmnt_awikner.classes import RunOpts, NumericalModel, Reservoir, ResOutput

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


@njit(fastmath=True)
def sum_numba_axis0(mat):
    # Computes the sum over axis 0 in numba compiled functions
    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.sum(mat[:, i])
    return res


@njit(fastmath=True)
def numba_var_axis0(pred):
    mean_all = sum_numba_axis0(pred) / pred.shape[0]
    variances_all = sum_numba_axis0((pred - mean_all) ** 2.0) / (pred.shape[0])
    return variances_all


@njit(fastmath=True)
def wasserstein_distance_empirical(measured_samples, true_samples):
    # Computes the wasserstein distance between the empirical CDFs of the two input sets of samples. Faster than scipy when compiled.
    if np.any(np.isnan(measured_samples)):
        return np.NAN
    if np.any(np.isinf(measured_samples)):
        return np.inf
    measured_samples.sort()
    true_samples.sort()
    n, m, n_inv, m_inv = (measured_samples.size, true_samples.size,
                          1 / measured_samples.size, 1 / true_samples.size)
    n_itr = 0
    m_itr = 0
    measured_cdf = 0
    true_cdf = 0
    wass_dist = 0
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
        measured_cdf += n_inv
        true_cdf += m_inv
        n_itr += 1
        m_itr += 1
    while n_itr < n and m_itr < m:
        if measured_samples[n_itr] < true_samples[m_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (measured_samples[n_itr] - prev_sample))
            prev_sample = measured_samples[n_itr]
            measured_cdf += n_inv
            n_itr += 1
        elif true_samples[m_itr] < measured_samples[n_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (true_samples[m_itr] - prev_sample))
            prev_sample = true_samples[m_itr]
            true_cdf += m_inv
            m_itr += 1
        else:
            wass_dist += np.abs((measured_cdf - true_cdf)
                                * (true_samples[m_itr] - prev_sample))
            prev_sample = true_samples[m_itr]
            measured_cdf += n_inv
            true_cdf += m_inv
            n_itr += 1
            m_itr += 1
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


@jit(fastmath=True, nopython=True)
def numba_eigsh(A):
    np.random.seed(0)
    v0 = np.random.rand(A.shape[0])
    with objmode(eigs_out='double[:]'):
        eigs_out = eigsh(A, k=6, v0=v0, maxiter=1e5, return_eigenvectors=False)
    return eigs_out


@jit(nopython=True, fastmath=True)
def numerical_model_wrapped(tau=0.1, T=300, ttsplit=5000, u0=0, system='lorenz', int_step = 1,
                            noise = np.zeros((1,1), dtype = np.double), params=np.array([[], []], dtype=np.complex128)):
    # Numba function for obtaining training and testing dynamical system time series data
    if system == 'lorenz':
        u_arr = np.ascontiguousarray(lorenzrungekutta(u0, T, tau, int_step))

        u_arr[0] = (u_arr[0] - 0) / 7.929788629895004
        u_arr[1] = (u_arr[1] - 0) / 8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896) / 8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict(u0, tau=tau, T=T, params=params, noise = noise)
        u_arr = np.ascontiguousarray(u_arr) / (1.1876770355823614)
    elif system == 'KS_d2175':
        u_arr, new_params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params, noise = noise)
        u_arr = np.ascontiguousarray(u_arr) / (1.2146066380280796)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit + 1]
    # u[ttsplit], the (ttsplit+1)st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params


@jit(nopython=True, fastmath=True)
def numerical_model_wrapped_pred(tau=0.1, T=300, ttsplit=5000, u0_array=np.array([[], []], dtype=np.complex128),
                             system='lorenz', int_step = 1, noise = np.zeros((1,1), dtype = np.double),
                             params=np.array([[], []], dtype=np.complex128)):
    # Numba function for obtaining training and testing dynamical system time series data for a set of initial conditions.
    # This is used during test to compute the map error instead of a for loop over the entire prediction period.
    if system == 'lorenz':
        u_arr = np.ascontiguousarray(
            lorenzrungekutta_pred(u0_array, tau, int_step))

        u_arr[0] = (u_arr[0] - 0) / 7.929788629895004
        u_arr[1] = (u_arr[1] - 0) / 8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896) / 8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict_pred(
            u0_array, tau=tau, T=T, params=params)
        u_arr = np.ascontiguousarray(u_arr) / (1.1876770355823614)
    elif system == 'KS_d2175':
        u_arr, new_params = kursiv_predict_pred(u0_array, tau=tau, T=T, params=params, d=21.75)
        u_arr = np.ascontiguousarray(u_arr) / (1.2146066380280796)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit + 1]
    # u[ttsplit], the (ttsplit+1)st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params


def getX(res, rk, x0=1, y0=1, z0=1):
    # Function to obtain reservoir states when in python interpreter. Calls get_X_wrapped.
    u_training = rk.u_arr_train
    res.X = get_X_wrapped(np.ascontiguousarray(u_training), res.X, res.Win_data, res.Win_indices, res.Win_indptr,
                          res.Win_shape,
                          res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage)

    return res.X


@jit(nopython=True, fastmath=True)
def get_X_wrapped(u_training, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
                  leakage, noise, noisetype='none', noise_scaling=0, noise_realization=0, traintype='normal'):
    # Numba compatible function for obtaining reservoir states using various types of noise.
    # Generally returns an array of reservoir states and the noiseless training data used as input.
    # If traintype is gradient, this function instead returns the resevoir states and the reservoir states derivatives.

    if noisetype in ['gaussian', 'perturbation']:
        if traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq'] or 'confined' in traintype:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i] + noise[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape,
                                                                        res_X[:, i]))
            u_training_wnoise = u_training + noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape,
                                                          res_X[:, i] + noise[:, i]))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]) + noise[:,
                                                                                                               i])
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
            res_X_nonoise[:, i + 1] = (1.0 - leakage) * res_X_nonoise[:, i] + leakage * np.tanh(
                mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        if traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq']:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size - noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0 - leakage) * res_X_nonoise[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i] + noise[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage) * temp_x + leakage * np.tanh(
                        mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                            1., u_training[:, i + k] + noise[:, i + k])) + mult_vec(W_data, W_indices, W_indptr,
                                                                                    W_shape, temp_x))
                res_X[:, i + noise_steps] = temp_x
            u_training_wnoise = u_training + noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size - noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0 - leakage) * res_X_nonoise[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x + noise[:, i]))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage) * temp_x + leakage * np.tanh(
                        mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                            1., u_training[:, i + k])) + mult_vec(W_data, W_indices, W_indptr, W_shape,
                                                                  temp_x + noise[:, i + k]))
                res_X[:, i + noise_steps] = temp_x
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size - noise_steps):
                temp_x = res_X_nonoise[:, i]
                temp_x = (1.0 - leakage) * res_X_nonoise[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x) + noise[:, i])
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage) * temp_x + leakage * np.tanh(
                        mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                            1., u_training[:, i + k])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x) + noise[
                                                                                                                  :,
                                                                                                                  i + k])
                res_X[:, i + noise_steps] = temp_x
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif noisetype in ['gaussian_onestep', 'perturbation_onestep']:
        if traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq']:
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * temp_x + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i] + noise[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                temp_x = np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i])) + mult_vec(
                        W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training + noise
        elif traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * temp_x + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x + noise[:, i]))
                temp_x = np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i])) + mult_vec(
                        W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        elif traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            # noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            temp_x = res_X[:, 0]
            for i in range(0, u_training[0].size):
                res_X[:, i + 1] = (1.0 - leakage) * temp_x + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x) + noise[:, i])
                temp_x = np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1., u_training[:, i])) + mult_vec(
                        W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif 'gaussianmult' in noisetype:
        if traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq'] or 'confined' in traintype:
            mean_input_var = np.sqrt(mean_numba_axis1(u_training**2.0))
            # noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_gen, noise_realization)
            for i in range(0, u_training[0].size):
                if 'varmean' in noisetype:
                    noise_scaling = mean_input_var
                else:
                    noise_scaling = u_training[:,i]
                if 'statemean' in noisetype:
                    noise_scaling = np.sqrt(np.mean(noise_scaling**2.0))*np.ones(noise_scaling.size)
                res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + leakage * np.tanh(
                    mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                        1., u_training[:, i] + noise_scaling*noise[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape,
                                                                        res_X[:, i]))
            u_training_wnoise = u_training + noise
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif 'gaussian' in noisetype:
        noise_steps = str_to_int(noisetype.replace('gaussian', ''))

        # noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_gen, noise_realization)
        res_X_nonoise = np.copy(res_X)
        for i in range(0, u_training[0].size):
            res_X_nonoise[:, i + 1] = (1.0 - leakage) * res_X_nonoise[:, i] + leakage * np.tanh(
                mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:, i]))
        for i in range(0, u_training[0].size - noise_steps):
            temp_x = res_X_nonoise[:, i]
            for k in range(noise_steps):
                if k == 0:
                    temp_x = (1.0 - leakage) * temp_x + leakage * np.tanh(
                        mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                            1., u_training[:, i + k] + noise[:, i + k])) + mult_vec(W_data, W_indices, W_indptr,
                                                                                    W_shape, temp_x))
                else:
                    temp_x = (1.0 - leakage) * temp_x + leakage * np.tanh(
                        mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                            1., u_training[:, i + k])) + mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            res_X[:, i + noise_steps] = temp_x
        u_training_wnoise = u_training + noise
        return res_X, u_training_wnoise

    elif traintype in ['sylvester_wD'] or 'gradient' in traintype or 'regzero' in traintype:
        rsvr_size = res_X.shape[0]
        res_D = np.zeros((rsvr_size, u_training.shape[1] + 1))
        for i in range(0, u_training[0].size):
            res_internal = mult_vec(Win_data, Win_indices, Win_indptr, Win_shape,
                                    np.append(1., u_training[:, i])) + mult_vec(
                W_data, W_indices, W_indptr, W_shape, res_X[:, i])
            res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + \
                              leakage * np.tanh(res_internal)
            res_D[:, i + 1] = leakage / (np.power(np.cosh(res_internal), 2.0))

        return res_X, res_D
    else:
        for i in range(0, u_training[0].size):
            res_X[:, i + 1] = (1.0 - leakage) * res_X[:, i] + leakage * np.tanh(
                mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(
                    1., u_training[:, i])) + mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:, i]))
        return res_X, u_training


def gen_noise_driver(run_opts, data_shape, res_shape, noise_scaling, noise_stream):
    # Generates an array of noise states for a given noise type, number of noise realizations, and array of random number generators
    if run_opts.noisetype in ['gaussian', 'perturbation'] or 'gaussianmult' in run_opts.noisetype:
        if run_opts.traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean',
                                  'rq'] or 'confined' in run_opts.traintype:
            noise = gen_noise(data_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        else:
            raise ValueError
    elif run_opts.noisetype not in ['gaussian_onestep', 'perturbation_onestep'] and \
            ('gaussian' in run_opts.noisetype and 'step' in run_opts.noisetype):
        if run_opts.traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq']:
            noise = gen_noise(data_shape[0], data_shape[1], str(
                run_opts.noisetype), noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        else:
            raise ValueError
    elif run_opts.noisetype in ['gaussian_onestep', 'perturbation_onestep']:
        if run_opts.traintype in ['normal', 'rmean', 'rplusq', 'rmeanq', 'rqmean', 'rq']:
            noise = gen_noise(data_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres1', 'rmeanres1', 'rplusqres1', 'rmeanqres1', 'rqmeanres1', 'rqres1']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        elif run_opts.traintype in ['normalres2', 'rmeanres2', 'rplusqres2', 'rmeanqres2', 'rqmeanres2', 'rqres2']:
            noise = gen_noise(res_shape[0], data_shape[1], run_opts.noisetype,
                              noise_scaling, noise_stream, run_opts.noise_realizations)
        else:
            raise ValueError
    elif 'gaussian' in run_opts.noisetype:
        noise = gen_noise(data_shape[0], data_shape[1], str(
            run_opts.noisetype), noise_scaling, noise_stream, run_opts.noise_realizations)
    elif 'gradient' in run_opts.noisetype or run_opts.noisetype == 'none':
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
                (noise_size, noise_length)) * np.sqrt(noise_scaling)
    if noisetype in ['perturbation', 'perturbation_onestep']:
        for noise_realization in noise_realizations:
            if noise_realization < noise_size:
                noise[noise_realization, noise_realization] = np.ones(
                    noise_length) * np.sqrt(noise_scaling)
            elif noise_realization < 2 * noise_length:
                noise[noise_realization, noise_realization -
                      noise_size] = -np.ones(noise_length) * np.sqrt(noise_scaling)
            else:
                raise ValueError

    return noise


def get_states(run_opts, res, rk, noise, noise_scaling=0):
    # Obtains the matrices used to train the reservoir using either linear regression or a sylvester equation
    # (in the case of Sylvester or Sylvester_wD training types)
    # Calls the numba compatible wrapped function
    if run_opts.traintype == 'getD':
        Dn = getD(np.ascontiguousarray(rk.u_arr_train), res.X, res.Win_data, res.Win_indices, res.Win_indptr,
                  res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
                  run_opts.discard_time, noise, run_opts.noisetype, noise_scaling, run_opts.noise_realizations,
                  run_opts.traintype, run_opts.squarenodes)
        return Dn
    elif run_opts.traintype in ['sylvester', 'sylvester_wD']:
        res.data_trstates, res.states_trstates, res.Y_train, res.X_train, res.left_mat = get_states_wrapped(
            np.ascontiguousarray(rk.u_arr_train), run_opts.reg_train_times, res.X, res.Win_data, res.Win_indices,
            res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            run_opts.discard_time, noise, run_opts.noisetype, noise_scaling, run_opts.noise_realizations,
            run_opts.traintype, run_opts.squarenodes)
    else:
        res.data_trstates, res.states_trstates, res.Y_train, res.X_train, res.gradient_reg = get_states_wrapped(
            np.ascontiguousarray(rk.u_arr_train), run_opts.reg_train_times, res.X, res.Win_data, res.Win_indices,
            res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,
            run_opts.discard_time, noise, run_opts.noisetype, noise_scaling, run_opts.noise_realizations,
            run_opts.traintype, run_opts.squarenodes)


@jit(nopython=True, fastmath=True)
def get_squared(X, rsvr_size, squarenodes, dim=0):
    X_aug = np.copy(X)
    if not squarenodes:
        return X_aug
    else:
        X_out = np.vstack(
            (X_aug[0].reshape(1, -1), X_aug[1:rsvr_size + 1], X_aug[1:rsvr_size + 1] ** 2.0, X_aug[rsvr_size + 1:]))
        return X_out


@jit(nopython=True, fastmath=True)
def get_squared_vec(X, rsvr_size, squarenodes):
    X_aug = np.copy(X)
    if not squarenodes:
        return X_aug
    else:
        X_out = np.concatenate(
            (np.array([X_aug[0]]), X_aug[1:rsvr_size + 1], X_aug[1:rsvr_size + 1] ** 2.0, X_aug[rsvr_size + 1:]))
        return X_out


@jit(nopython=True, fastmath=True)
def get_states_wrapped(u_arr_train, reg_train_times, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data,
                       W_indices, W_indptr, W_shape, leakage, skip, noise, noisetype='none',
                       noise_scaling=0, noise_realizations=1, traintype='normal', squarenodes=False, q=0):
    # Numba compatible function to obtain the matrices used to train the reservoir using either linear regression or a sylvester equation.
    # The type of matrices depends on the traintype, number of noise realizations, and noisetype
    res_X = np.ascontiguousarray(res_X)
    u_arr_train = np.ascontiguousarray(u_arr_train)
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    if squarenodes:
        res_feature_size = 2 * rsvr_size
    else:
        res_feature_size = rsvr_size
    data_trstates = np.zeros((n, res_feature_size + n + 1))
    states_trstates = np.zeros((reg_train_times.size, res_feature_size + n + 1, res_feature_size + n + 1))
    gradient_reg = np.zeros((reg_train_times.size, res_feature_size + n + 1, res_feature_size + n + 1))
    Y_train = np.ascontiguousarray(u_arr_train[:, skip:-1])
    print('Reg train times:')
    print(reg_train_times)
    reg_train_fracs = 1.0 / reg_train_times
    print('Reg train fracs:')
    print(reg_train_fracs)
    if traintype in ['normal', 'normalres1', 'normalres2']:
        # Normal multi-noise training that sums all reservoir state outer products
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = X[:, skip:(res_d - 2)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            data_trstates += Y_train @ X_train.T
            states_trstates[0] += X_train @ X_train.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rmeanq', 'rmeanqres1', 'rmeanqres2']:
        # Training using the mean and the rescaled sum of the perturbations from the mean
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
                X_all = np.zeros((X_train.shape[0], X_train.shape[1], noise_realizations))
            X_train_mean += X_train / noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_mean.T
        states_trstates[0] = X_train_mean @ X_train_mean.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        for (j, reg_train_time), prev_reg_train_time in zip(enumerate(reg_train_times),
                                                            np.append(np.zeros(1, dtype=np.int64),
                                                                      reg_train_times[:-1])):
            for i in range(noise_realizations):
                Q_fit = X_all[:, prev_reg_train_time:reg_train_time, i] - \
                        X_train_mean[:, prev_reg_train_time:reg_train_time]
                Q_fit_2 = Q_fit @ Q_fit.T
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += Q_fit_2 * reg_train_fracs[k] / noise_realizations
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rmean', 'rmeanres1', 'rmeanres2']:
        # Training using the mean only
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = X[:, skip:(res_d - 2)]
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train / noise_realizations
        data_trstates = Y_train @ X_train_mean.T
        states_trstates[0] = X_train_mean @ X_train_mean.T
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rqmean', 'rqmeanres1', 'rqmeanres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the mean
        X_0, u_arr_train_noise = get_X_wrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
            leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip:(res_d - 2)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d - (skip + 1))), X_0, u_arr_train[:, skip - 1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train / noise_realizations
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates[0] = X_train_0 @ X_train_0.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        prev_reg_train_time = 0
        for j, reg_train_time in enumerate(reg_train_times):
            for i in range(noise_realizations):
                Q_fit = X_all[:, prev_reg_train_time:reg_train_time, i] - \
                        X_train_mean[:, prev_reg_train_time:reg_train_time]
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += (Q_fit @ Q_fit.T) * reg_train_fracs[k] / noise_realizations
            prev_reg_train_time = reg_train_time
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rq', 'rqres1', 'rqres2']:
        # Training using the noiseless reservoir and the rescaled perturbations from the noiseless reservoir
        X_0, u_arr_train_noise = get_X_wrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
            leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip:(res_d - 2)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d - (skip + 1))), X_0, u_arr_train[:, skip - 1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            X_all[:, :, i] = X_train
        data_trstates = Y_train @ X_train_0.T
        states_trstates[0] = X_train_0 @ X_train_0.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.copy(states_trstates[0])
        prev_reg_train_time = 0
        for j, reg_train_time in enumerate(reg_train_times):
            for i in range(noise_realizations):
                Q_fit = X_all[:, prev_reg_train_time:reg_train_time, i] - \
                        X_train_0[:, prev_reg_train_time:reg_train_time]
                for k in range(j, reg_train_times.size):
                    states_trstates[k] += (Q_fit @ Q_fit.T) * reg_train_fracs[k] / noise_realizations
            prev_reg_train_time = reg_train_time
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif traintype in ['rplusq', 'rplusqres1', 'rplusqres2']:
        # Training using the sum of the outer products of the sum of the noiseless reservoir and
        # the perturbation from the mean
        X_0, u_arr_train_noise = get_X_wrapped(
            u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
            leakage, noise[0])
        X_0 = np.ascontiguousarray(X_0[:, skip:(res_d - 2)])
        X_train_0 = np.ascontiguousarray(np.concatenate(
            (np.ones((1, d - (skip + 1))), X_0, u_arr_train[:, skip - 1:-2]), axis=0))
        X_train_0 = get_squared(X_train_0, rsvr_size, squarenodes)
        X_all = np.zeros(
            (X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                                 W_data, W_indices, W_indptr, W_shape, leakage, noise[i], noisetype,
                                                 noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip:(res_d - 2)])
            X_train = np.ascontiguousarray(np.concatenate(
                (np.ones((1, d - (skip + 1))), X, u_arr_train_noise[:, skip - 1:-2]), axis=0))
            X_train = get_squared(X_train, rsvr_size, squarenodes)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train / noise_realizations
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
    elif 'gradientk' in traintype and 'only' not in traintype and 'mult' not in traintype:
        # Linearized k-step noise
        k = str_to_int(traintype.replace('gradientk', ''))
        reg_train_fracs = 1.0 / (reg_train_times - (k - 1))
        sparse_cutoff = 0.89
        break_flag = False
        X, D = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                             W_indptr,
                             W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d - (skip + 1))), X, u_arr_train[:, skip - 1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        gradient_reg_base = np.zeros((res_feature_size + n + 1, res_feature_size + n + 1))
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
            get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
        D_n_datas = List()
        D_n_indices = List()
        D_n_indptrs = List()
        D_n_shape = np.array([res_feature_size + n + 1, n])
        E_n_datas = List()
        E_n_indices = List()
        E_n_indptrs = List()
        E_n_shape = np.array([res_feature_size + n + 1, res_feature_size + n + 1])
        reg_comp_datas = List()
        reg_comp_indices = List()
        reg_comp_indptrs = List()
        reg_comp_shape = np.array([res_feature_size + n + 1, n])
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n,
            squarenodes)
        leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage,
                                                                                             squarenodes)
        reg_sum_avg_runtime = 0.
        E_n_avg_runtime = 0.
        reg_mult_avg_runtime = 0.
        D_n_avg_runtime = 0.

        for i in range(k):
            D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:, i], X[:, i], Win_nobias_data, Win_nobias_indices,
                                                    Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size,
                                                    res_feature_size, n, squarenodes)
            D_n_datas.append(np.ascontiguousarray(D_n_data))
            D_n_indices.append(np.ascontiguousarray(D_n_idx))
            D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
        if k > 1:
            for i in range(1, k):
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i], E_n_shape, rsvr_size, W_mat_data,
                                                        W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data,
                                                        leakage_indices,
                                                        leakage_indptr, leakage_shape, squarenodes)
                E_n_datas.append(np.ascontiguousarray(E_n_data))
                E_n_indices.append(np.ascontiguousarray(E_n_idx))
                E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))

        for i in range(k - 1):
            reg_comp_data, reg_comp_idx, reg_comp_indptr = \
                np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
            if k > 1:
                for j in range(i, k - 1):
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                        E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
            reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
            reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
            reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
        reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
        reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
        reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
        sparsity = np.array([reg_comp_datas[j].size / (reg_comp_shape[0] * reg_comp_shape[1]) for j in range(k)])

        for i in range(k, X.shape[1]):
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = (i == reg_train_times)
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                print(gradient_reg[assign_grad_reg].shape)
                print((gradient_reg_base * reg_train_fracs[assign_grad_reg]).shape)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
            if assign_grad_reg[-1]:
                break_flag = True
                break
            if k > 1:
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i],
                                                        E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr,
                                                        W_mat_shape,
                                                        leakage_data, leakage_indices, leakage_indptr, leakage_shape,
                                                        squarenodes)
                E_n_datas[k - 2] = np.ascontiguousarray(E_n_data)
                E_n_indices[k - 2] = np.ascontiguousarray(E_n_idx)
                E_n_indptrs[k - 2] = np.ascontiguousarray(E_n_indptr)
            for j in range(k - 1):
                if k > 1:
                    if sparsity[j + 1] < sparse_cutoff:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    else:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                        np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                        np.ascontiguousarray(reg_comp_indptr)
            reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:, i], X[:, i],
                                                                   Win_nobias_data, Win_nobias_indices,
                                                                   Win_nobias_indptr, Win_nobias_shape, D_n_shape,
                                                                   rsvr_size, res_feature_size, n, squarenodes)
            reg_comp_datas[k - 1], reg_comp_indices[k - 1], reg_comp_indptrs[k - 1] = \
                np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                np.ascontiguousarray(reg_comp_indptr)
        if not break_flag:
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = i + 1 == reg_train_times
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.ascontiguousarray(states_trstates[0])
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif 'gradientmult' in traintype:
        if 'varmean' in traintype and 'statemean' not in traintype:
            k = str_to_int(traintype.replace('gradientmult_varmeank', ''))
            U = np.sqrt(mean_numba_axis1(u_arr_train**2.0)).reshape(-1,1) @ \
                np.ones((1,u_arr_train[:, skip - 1:-2].shape[1]))
        elif 'varmean' not in traintype and 'statemean' in traintype:
            k = str_to_int(traintype.replace('gradientmult_statemeank', ''))
            U = np.ones((n,1)) @ np.sqrt(sum_numba_axis0(u_arr_train**2.0)/n).reshape(1,-1)
        elif 'varmean' in traintype and 'statemean' in traintype:
            k = str_to_int(traintype.replace('gradientmult_varmean_statemeank', ''))
            U = np.ones(u_arr_train[:, skip - 1:-2].shape) * np.sqrt(np.mean(u_arr_train**2.0))
        else:
            k = str_to_int(traintype.replace('gradientmultk', ''))
            U = u_arr_train[:, skip - 1:-2]
            print('Gradient mult k:')
            print(k)
        # Linearized k-step noise
        # k = str_to_int(traintype.replace('gradientk', ''))
        reg_train_fracs = 1.0 / (reg_train_times - (k - 1))
        sparse_cutoff = 0.89
        break_flag = False
        X, D = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                             W_indptr,
                             W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d - (skip + 1))), X, u_arr_train[:, skip - 1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        gradient_reg_base = np.zeros((res_feature_size + n + 1, res_feature_size + n + 1))
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
            get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
        D_n_datas = List()
        D_n_indices = List()
        D_n_indptrs = List()
        D_n_shape = np.array([res_feature_size + n + 1, n])
        E_n_datas = List()
        E_n_indices = List()
        E_n_indptrs = List()
        E_n_shape = np.array([res_feature_size + n + 1, res_feature_size + n + 1])
        reg_comp_datas = List()
        reg_comp_indices = List()
        reg_comp_indptrs = List()
        reg_comp_shape = np.array([res_feature_size + n + 1, n])
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n,
            squarenodes)
        leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage,
                                                                                             squarenodes)
        reg_sum_avg_runtime = 0.
        E_n_avg_runtime = 0.
        reg_mult_avg_runtime = 0.
        D_n_avg_runtime = 0.

        for i in range(k):
            D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:, i], X[:, i], Win_nobias_data, Win_nobias_indices,
                                                    Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size,
                                                    res_feature_size, n, squarenodes)
            D_n_datas.append(np.ascontiguousarray(D_n_data))
            D_n_indices.append(np.ascontiguousarray(D_n_idx))
            D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
        if k > 1:
            for i in range(1, k):
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i], E_n_shape, rsvr_size, W_mat_data,
                                                        W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data,
                                                        leakage_indices,
                                                        leakage_indptr, leakage_shape, squarenodes)
                E_n_datas.append(np.ascontiguousarray(E_n_data))
                E_n_indices.append(np.ascontiguousarray(E_n_idx))
                E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))

        for i in range(k - 1):
            reg_comp_data, reg_comp_idx, reg_comp_indptr = \
                np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
            if k > 1:
                for j in range(i, k - 1):
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                        E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
            reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
            reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
            reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
        reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
        reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
        reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
        sparsity = np.array([reg_comp_datas[j].size / (reg_comp_shape[0] * reg_comp_shape[1]) for j in range(k)])

        for i in range(k, X.shape[1]):
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    gradient_reg_base += matrix_sparse_diag_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape,
                        U[:, i - (k - j)])
                else:
                    gradient_reg_base += matrix_sparse_diag_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape,
                                                                    U[:, i - (k - j)])
            assign_grad_reg = (i == reg_train_times)
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                print(gradient_reg[assign_grad_reg].shape)
                print((gradient_reg_base * reg_train_fracs[assign_grad_reg]).shape)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
            if assign_grad_reg[-1]:
                break_flag = True
                break
            if k > 1:
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i],
                                                        E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr,
                                                        W_mat_shape,
                                                        leakage_data, leakage_indices, leakage_indptr, leakage_shape,
                                                        squarenodes)
                E_n_datas[k - 2] = np.ascontiguousarray(E_n_data)
                E_n_indices[k - 2] = np.ascontiguousarray(E_n_idx)
                E_n_indptrs[k - 2] = np.ascontiguousarray(E_n_indptr)
            for j in range(k - 1):
                if k > 1:
                    if sparsity[j + 1] < sparse_cutoff:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    else:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                        np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                        np.ascontiguousarray(reg_comp_indptr)
            reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:, i], X[:, i],
                                                                   Win_nobias_data, Win_nobias_indices,
                                                                   Win_nobias_indptr, Win_nobias_shape, D_n_shape,
                                                                   rsvr_size, res_feature_size, n, squarenodes)
            reg_comp_datas[k - 1], reg_comp_indices[k - 1], reg_comp_indptrs[k - 1] = \
                np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                np.ascontiguousarray(reg_comp_indptr)
        if not break_flag:
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    gradient_reg_base += matrix_sparse_diag_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape,
                        U[:, i - (k - j)])
                else:
                    gradient_reg_base += matrix_sparse_diag_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape,
                                                                    U[:, i - (k - j)])
            assign_grad_reg = i + 1 == reg_train_times
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.ascontiguousarray(states_trstates[0])
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    elif 'gradientk' in traintype and 'only' in traintype:
        # Linearized k-step noise
        k = str_to_int(traintype.replace('gradientk', '').replace('only', ''))
        reg_train_fracs = 1.0 / (reg_train_times - (k - 1))
        sparse_cutoff = 0.89
        break_flag = False
        X, D = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                             W_indptr,
                             W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        D = D[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d - (skip + 1))), X, u_arr_train[:, skip - 1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        gradient_reg_base = np.zeros((res_feature_size + n + 1, res_feature_size + n + 1))
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
            get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
        D_n_datas = List()
        D_n_indices = List()
        D_n_indptrs = List()
        D_n_shape = np.array([res_feature_size + n + 1, n])
        E_n_datas = List()
        E_n_indices = List()
        E_n_indptrs = List()
        E_n_shape = np.array([res_feature_size + n + 1, res_feature_size + n + 1])
        reg_comp_datas = List()
        reg_comp_indices = List()
        reg_comp_indptrs = List()
        reg_comp_shape = np.array([res_feature_size + n + 1, n])
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n,
            squarenodes)
        leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage,
                                                                                             squarenodes)
        reg_sum_avg_runtime = 0.
        E_n_avg_runtime = 0.
        reg_mult_avg_runtime = 0.
        D_n_avg_runtime = 0.

        for i in range(k):
            D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:, i], X[:, i], Win_nobias_data, Win_nobias_indices,
                                                    Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size,
                                                    res_feature_size, n, squarenodes)
            D_n_datas.append(np.ascontiguousarray(D_n_data))
            D_n_indices.append(np.ascontiguousarray(D_n_idx))
            D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
        if k > 1:
            for i in range(1, k):
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i], E_n_shape, rsvr_size, W_mat_data,
                                                        W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data,
                                                        leakage_indices,
                                                        leakage_indptr, leakage_shape, squarenodes)
                E_n_datas.append(np.ascontiguousarray(E_n_data))
                E_n_indices.append(np.ascontiguousarray(E_n_idx))
                E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))

        for i in range(k - 1):
            reg_comp_data, reg_comp_idx, reg_comp_indptr = \
                np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
            if k > 1:
                for j in range(i, k - 1):
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                        E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
            reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
            reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
            reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
        reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
        reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
        reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
        sparsity = np.array([reg_comp_datas[j].size / (reg_comp_shape[0] * reg_comp_shape[1]) for j in range(k)])

        for i in range(k, X.shape[1]):
            if sparsity[j] < sparse_cutoff:
                gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                    reg_comp_datas[0], reg_comp_indices[0], reg_comp_indptrs[0], reg_comp_shape)
            else:
                gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[0], reg_comp_indices[0],
                                                                reg_comp_indptrs[0], reg_comp_shape)
            assign_grad_reg = (i == reg_train_times)
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                print(gradient_reg[assign_grad_reg].shape)
                print((gradient_reg_base * reg_train_fracs[assign_grad_reg]).shape)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
            if assign_grad_reg[-1]:
                break_flag = True
                break
            if k > 1:
                E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i],
                                                        E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr,
                                                        W_mat_shape,
                                                        leakage_data, leakage_indices, leakage_indptr, leakage_shape,
                                                        squarenodes)
                E_n_datas[k - 2] = np.ascontiguousarray(E_n_data)
                E_n_indices[k - 2] = np.ascontiguousarray(E_n_idx)
                E_n_indptrs[k - 2] = np.ascontiguousarray(E_n_indptr)
            for j in range(k - 1):
                if k > 1:
                    if sparsity[j + 1] < sparse_cutoff:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    else:
                        reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                            E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                            reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                    reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                        np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                        np.ascontiguousarray(reg_comp_indptr)
            reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:, i], X[:, i],
                                                                   Win_nobias_data, Win_nobias_indices,
                                                                   Win_nobias_indptr, Win_nobias_shape, D_n_shape,
                                                                   rsvr_size, res_feature_size, n, squarenodes)
            reg_comp_datas[k - 1], reg_comp_indices[k - 1], reg_comp_indptrs[k - 1] = \
                np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                np.ascontiguousarray(reg_comp_indptr)
        if not break_flag:
            if sparsity[j] < sparse_cutoff:
                gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                    reg_comp_datas[0], reg_comp_indices[0], reg_comp_indptrs[0], reg_comp_shape)
            else:
                gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[0], reg_comp_indices[0],
                                                                reg_comp_indptrs[0], reg_comp_shape)
            assign_grad_reg = i + 1 == reg_train_times
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.ascontiguousarray(states_trstates[0])
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
        """
    elif traintype == 'sylvester':
        # Sylvester regularization w/o derivative
        X, p = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
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
        X, D = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
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
        """
    elif 'regzero' in traintype:
        # Linearized k-step noise
        k = str_to_int(traintype.replace('regzerok', ''))
        reg_train_fracs = 1.0 / (reg_train_times - (k - 1))
        sparse_cutoff = 0.89
        break_flag = False
        X, tmp = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                               W_indptr,
                               W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X[:, skip:(res_d - 2)]
        X_train = np.concatenate(
            (np.ones((1, d - (skip + 1))), X, u_arr_train[:, skip - 1:-2]), axis=0)
        X_train = get_squared(X_train, rsvr_size, squarenodes)
        data_trstates = Y_train @ X_train.T
        states_trstates[0] = X_train @ X_train.T
        for i in range(1, reg_train_times.size):
            states_trstates[i] = np.ascontiguousarray(states_trstates[0])
        X_zero, D_zero = get_X_wrapped(np.zeros(u_arr_train.shape), res_X, Win_data, Win_indices, Win_indptr, Win_shape,
                                       W_data, W_indices, W_indptr,
                                       W_shape, leakage, noise[0], noisetype, noise_scaling, 1, traintype)
        X = X_zero[:, skip:(res_d - 2)]
        D = D_zero[:, skip:(res_d - 2)]
        gradient_reg_base = np.zeros((res_feature_size + n + 1, res_feature_size + n + 1))
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
            get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
        D_n_datas = List()
        D_n_indices = List()
        D_n_indptrs = List()
        D_n_shape = np.array([res_feature_size + n + 1, n])
        E_n_datas = List()
        E_n_indices = List()
        E_n_indptrs = List()
        E_n_shape = np.array([res_feature_size + n + 1, res_feature_size + n + 1])
        reg_comp_datas = List()
        reg_comp_indices = List()
        reg_comp_indptrs = List()
        reg_comp_shape = np.array([res_feature_size + n + 1, n])
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
            Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n,
            squarenodes)
        leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage,
                                                                                             squarenodes)
        reg_sum_avg_runtime = 0.
        E_n_avg_runtime = 0.
        reg_mult_avg_runtime = 0.
        D_n_avg_runtime = 0.

        for i in range(k):
            D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:, i], X[:, i], Win_nobias_data, Win_nobias_indices,
                                                    Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size,
                                                    res_feature_size, n, squarenodes)
            D_n_datas.append(np.ascontiguousarray(D_n_data))
            D_n_indices.append(np.ascontiguousarray(D_n_idx))
            D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
        for i in range(1, k):
            E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i], E_n_shape, rsvr_size, W_mat_data,
                                                    W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data,
                                                    leakage_indices,
                                                    leakage_indptr, leakage_shape, squarenodes)
            E_n_datas.append(np.ascontiguousarray(E_n_data))
            E_n_indices.append(np.ascontiguousarray(E_n_idx))
            E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))

        for i in range(k - 1):
            reg_comp_data, reg_comp_idx, reg_comp_indptr = \
                np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
            for j in range(i, k - 1):
                reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                    E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
            reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
            reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
            reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
        reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
        reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
        reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
        sparsity = np.array([reg_comp_datas[j].size / (reg_comp_shape[0] * reg_comp_shape[1]) for j in range(k)])

        for i in range(k, X.shape[1]):
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = (i == reg_train_times)
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                print(gradient_reg[assign_grad_reg].shape)
                print((gradient_reg_base * reg_train_fracs[assign_grad_reg]).shape)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
            if assign_grad_reg[-1]:
                break_flag = True
                break
            E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:, i], X[:, i],
                                                    E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr,
                                                    W_mat_shape,
                                                    leakage_data, leakage_indices, leakage_indptr, leakage_shape,
                                                    squarenodes)
            E_n_datas[k - 2] = np.ascontiguousarray(E_n_data)
            E_n_indices[k - 2] = np.ascontiguousarray(E_n_idx)
            E_n_indptrs[k - 2] = np.ascontiguousarray(E_n_indptr)

            for j in range(k - 1):
                if sparsity[j + 1] < sparse_cutoff:
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(
                        E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                        reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                else:
                    reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(
                        E_n_datas[k - 2], E_n_indices[k - 2], E_n_indptrs[k - 2], E_n_shape,
                        reg_comp_datas[j + 1], reg_comp_indices[j + 1], reg_comp_indptrs[j + 1], reg_comp_shape)
                reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                    np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                    np.ascontiguousarray(reg_comp_indptr)
            reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:, i], X[:, i],
                                                                   Win_nobias_data, Win_nobias_indices,
                                                                   Win_nobias_indptr, Win_nobias_shape, D_n_shape,
                                                                   rsvr_size, res_feature_size, n, squarenodes)
            reg_comp_datas[k - 1], reg_comp_indices[k - 1], reg_comp_indptrs[k - 1] = \
                np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx), \
                np.ascontiguousarray(reg_comp_indptr)
        if not break_flag:
            for j in range(k):
                if sparsity[j] < sparse_cutoff:
                    # gradient_reg += reg_components[:, :, j] @ reg_components[:, :, j].T
                    gradient_reg_base += matrix_sparse_sparseT_conv_mult(
                        reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
                else:
                    gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],
                                                                    reg_comp_indptrs[j], reg_comp_shape)
            assign_grad_reg = i + 1 == reg_train_times
            if np.any(assign_grad_reg):
                print(assign_grad_reg)
                gradient_reg[assign_grad_reg] = gradient_reg_base * reg_train_fracs[assign_grad_reg]
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg
    else:
        # Noiseless training
        data_trstates = np.zeros((n, res_feature_size + 1 + n), dtype=np.float64)
        states_trstates[0] = np.zeros(
            (n + res_feature_size + 1, n + res_feature_size + 1), dtype=np.float64)
        X_train = np.zeros((n + res_feature_size + 1, d - (skip + 1)))
        return data_trstates, states_trstates, Y_train, X_train, gradient_reg


@jit(nopython=True, fastmath=True)
def getD(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape,
         leakage, noise, skip, noisetype='none',
         noise_scaling=0, noise_realizations=1, traintype='normal', squarenodes=False):
    n, d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    Y_train = u_arr_train[:, skip + 1:]
    X, D = get_X_wrapped(u_arr_train, res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices,
                         W_indptr, W_shape, leakage, noise, 'none', 0, 0, 'gradient', squarenodes)
    return [D[:, skip:(res_d - 2)]]


# CONCATENATE ALL THE DATA BEFORE RUNNING REGRESSION


def predict(res, u0, steps=1000, squarenodes=False):
    # Wrapper for the prediction function
    Y = predictwrapped(res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices,
                       res.W_indptr, res.W_shape, res.Wout, res.leakage, u0, steps, squarenodes)
    return Y


@jit(nopython=True, fastmath=True)
def predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, Wout,
                   leakage, u0, steps, squarenodes=False):
    # Numba compatible prediction function
    Y = np.empty((Win_shape[1] - 1, steps + 1))
    X = np.empty((res_X.shape[0], steps + 1))

    Y[:, 0] = u0
    X[:, 0] = res_X[:, -2]

    for i in range(0, steps):
        X[:, i + 1] = (1 - leakage) * X[:, i] + leakage * np.tanh(
            mult_vec(Win_data, Win_indices, Win_indptr, Win_shape, np.append(1.,
                                                                             Y[:, i])) + mult_vec(W_data, W_indices,
                                                                                                  W_indptr, W_shape,
                                                                                                  X[:, i]))
        Y[:, i + 1] = Wout @ get_squared_vec(np.concatenate((np.array([1.]), X[:, i + 1], Y[:, i])),
                                             X.shape[0], squarenodes)

    return Y


def get_test_data(run_opts, test_stream, dnoise_test_stream, overall_idx, rkTime, split):
    # Function for obtaining test data sets used to validate reservoir performance
    # Uses an array of random number generators
    if run_opts.system == 'lorenz':
        ic = test_stream[0].random(3) * 2 - 1
        u0 = np.array([ic[0], ic[1], 30*ic[2]])
        int_step = int(run_opts.tau / 0.01)
    elif run_opts.system in ['KS', 'KS_d2175']:
        u0 = (test_stream[0].random(64) * 2 - 1) * 0.6
        u0 = u0 - np.mean(u0)
        int_step = 1
    transient = 2000
    total_iterations = rkTime + transient + split
    dnoise = dnoise_test_stream[0].standard_normal((u0.size, total_iterations)) * np.sqrt(run_opts.dyn_noise)
    u_arr_train_nonoise, u_arr_test, p, params = numerical_model_wrapped(tau=run_opts.tau, T=total_iterations,
                                                                     ttsplit=split + transient,
                                                                     u0=u0, system=run_opts.system, int_step = int_step,
                                                                     noise  = dnoise)
    u_arr_train_nonoise = u_arr_train_nonoise[:, transient:]
    rktest_u_arr_train_nonoise = np.zeros(
        (u_arr_train_nonoise.shape[0], u_arr_train_nonoise.shape[1], run_opts.num_tests))
    rktest_u_arr_test = np.zeros(
        (u_arr_test.shape[0], u_arr_test.shape[1], run_opts.num_tests))
    rktest_u_arr_train_nonoise[:, :, 0] = u_arr_train_nonoise
    rktest_u_arr_test[:, :, 0] = u_arr_test
    """
    print('Test data %d' % 0)
    print(rktest_u_arr_test[-3:,-3:,0])
    """
    for i in range(1, run_opts.num_tests):
        # np.random.seed(i)
        if run_opts.system == 'lorenz':
            ic = test_stream[i].random(3) * 2 - 1
            u0 = np.array([ic[0],ic[1], 30 * ic[2]])
        elif run_opts.system in ['KS', 'KS_d2175']:
            u0 = (test_stream[i].random(64) * 2 - 1) * 0.6
            u0 = u0 - np.mean(u0)
        dnoise = dnoise_test_stream[i].standard_normal((u0.size, total_iterations * int_step)) * np.sqrt(run_opts.dyn_noise)
        u_arr_train_nonoise, rktest_u_arr_test[:, :, i], p, params = numerical_model_wrapped(tau=run_opts.tau,
                                                                                         T=rkTime + transient + split,
                                                                                         ttsplit=split + transient,
                                                                                         u0=u0, system=run_opts.system,
                                                                                         int_step = int_step,
                                                                                         params=params, noise = dnoise)
        rktest_u_arr_train_nonoise[:, :, i] = u_arr_train_nonoise[:, transient:]
        """
        print('Test data %d' % i)
        print(rktest_u_arr_test[-3:,-3:,i])
        """

    if run_opts.save_truth and overall_idx == 0:
        for (i, test) in enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)):
            np.savetxt(os.path.join(run_opts.run_folder_name,
                                    '%s_tau%0.2f_true_test_%d.csv' % (run_opts.system, run_opts.tau, test)),
                       rktest_u_arr_test[:, :, i], delimiter=',')

    return rktest_u_arr_train_nonoise, rktest_u_arr_test, params


def test(run_opts, res, Wout_itr, noise_in, rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max, rkTime=1000,
         split=3000, params=np.array([[], []], dtype=np.complex128), showMapError=False, showTrajectories=False,
         showHist=False):
    # Wrapper function for the numba compatible test function.

    stable_count, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist, pmap_max, pmap_max_wass_dist = test_wrapped(
        res.X, res.Win_data, res.Win_indices, res.Win_indptr, res.Win_shape, res.W_data, res.W_indices, res.W_indptr,
        res.W_shape, res.Wout[Wout_itr], res.leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max,
        run_opts.num_tests, rkTime, split, noise_in, run_opts.system, params=params, pmap=run_opts.pmap,
        max_valid_time=run_opts.max_valid_time, squarenodes=run_opts.squarenodes, savepred=run_opts.savepred,
        save_time_rms=run_opts.save_time_rms)

    return stable_count / run_opts.num_tests, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist, pmap_max, pmap_max_wass_dist


@jit(nopython=True, fastmath=True)
def test_wrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, Wout,
                 leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test, true_pmap_max, num_tests, rkTime, split,
                 noise_in, system='lorenz', tau=0.1, params=np.array([[], []], dtype=np.complex128), pmap=False,
                 max_valid_time=500, squarenodes=False, savepred=False, save_time_rms=False):
    # Numba compatable function for testing trained reservoir performance against true system time series
    stable_count = 0
    num_vt_tests = ((rkTime - split)) // max_valid_time
    valid_time = np.zeros((num_tests, num_vt_tests))
    max_rms = np.zeros(num_tests)
    mean_rms = np.zeros(num_tests)
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    wass_dist = np.zeros(num_tests)
    pmap_max = []
    pmap_max_wass_dist = np.zeros(num_tests)
    if savepred:
        preds = np.zeros((num_tests, rktest_u_arr_test.shape[0], (rkTime - split) + 1))
    else:
        preds = np.empty((1, 1, 1), dtype=np.double)
    if save_time_rms:
        rms = np.zeros((num_tests, (rkTime - split)))
    else:
        rms = np.empty((1, 1), dtype=np.double)

    # print(num_tests)
    for i in range(num_tests):
        with objmode(test_tic='double'):
            test_tic = time.perf_counter()
        res_X = (np.zeros((res_X.shape[0], split + 2)) * 2 - 1)

        # sets res.X
        res_X, p = get_X_wrapped(np.ascontiguousarray(
            rktest_u_arr_train_nonoise[:, :, i]), res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data,
            W_indices, W_indptr, W_shape, leakage, noise_in)
        pred_full = predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                                   W_shape,
                                   Wout, leakage, u0=rktest_u_arr_test[:, 0, i], steps=(rkTime - split),
                                   squarenodes=squarenodes)
        if savepred:
            preds[i] = pred_full
        error = np.zeros(max_valid_time)
        if pmap:
            if system == 'lorenz':
                calc_pred = np.stack((pred_full[0] * 7.929788629895004,
                                      pred_full[1] * 8.9932616136662,
                                      pred_full[2] * 8.575917849311919 + 23.596294463016896))
                # wass_dist[i] = wasserstein_distance_empirical(calc_pred.flatten(), true_trajectory.flatten())
                pred_pmap_max = poincare_max(calc_pred, np.arange(pred_full.shape[0]))
            elif system == 'KS':
                # wass_dist[i] = wasserstein_distance_empirical(pred.flatten()*1.1876770355823614, true_trajectory.flatten())
                pred_pmap_max = poincare_max(pred_full * 1.1876770355823614, np.arange(pred_full.shape[0]))
            elif system == 'KS_d2175':
                pred_pmap_max = poincare_max(pred_full * 1.2146066380280796, np.arange(pred_full.shape[0]))

            pmap_max.append(pred_pmap_max)
            for j in range(rktest_u_arr_test.shape[0]):
                if j == 0:
                    pred_pmap_max_all = pred_pmap_max[j]
            pmap_max_wass_dist[i] = wasserstein_distance_empirical(pred_pmap_max_all, true_pmap_max)
        else:
            pmap_max_wass_dist[i] = np.nan

        if system == 'KS':
            vt_cutoff = 0.2 * 1.3697994268693887
        else:
            vt_cutoff = 0.2 * np.sqrt(2)
        check_vt = True
        array_compute = False
        pred = pred_full[:, :max_valid_time]
        for k in range(num_vt_tests):
            for j in range(1, pred.shape[1]):
                error[j] = np.sqrt(
                    np.mean((pred[:, j] - rktest_u_arr_test[:, k * max_valid_time + j, i]) ** 2.0))

                if error[j] < vt_cutoff and check_vt:
                    valid_time[i, k] = j
                else:
                    # if check_vt:
                    check_vt = False
            print('Valid Time')
            print(valid_time[i, k])
            res_X = np.zeros((res_X.shape[0], max_valid_time + 2))
            res_X, p = get_X_wrapped(np.ascontiguousarray(
                rktest_u_arr_test[:, k * max_valid_time:(k + 1) * max_valid_time + 1, i]), res_X, Win_data, Win_indices,
                Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, leakage, noise_in)
            if k < (num_vt_tests - 1):
                pred = predictwrapped(res_X, Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr,
                                      W_shape,
                                      Wout, leakage, u0=rktest_u_arr_test[:, (k + 1) * max_valid_time, i],
                                      steps=max_valid_time - 1, squarenodes=squarenodes)
                check_vt = True
        if array_compute:
            if system == 'lorenz':
                int_step = math.trunc(tau / 0.01)
                rkmap_u_arr_train = numerical_model_wrapped_pred(u0_array=np.stack((pred_full[0] * 7.929788629895004,
                                                                pred_full[1] * 8.9932616136662,
                                                                pred_full[2] * 8.575917849311919 + 23.596294463016896)),
                                                             int_step = int_step, system=system, params=params, tau=tau,
                                                             ttsplit=pred_full.shape[1])[0]
            elif system == 'KS':
                u0 = pred_full * 1.1876770355823614
                rkmap_u_arr_train = numerical_model_wrapped_pred(
                    u0_array=u0, tau=tau, T=1, int_step = 1, system=system, params=params, ttsplit=pred_full.shape[1])[0]
            elif system == 'KS_d2175':
                u0 = pred_full * 1.2146066380280796
                rkmap_u_arr_train = numerical_model_wrapped_pred(
                    u0_array=u0, tau=tau, T=1, int_step = 1, system=system, params=params, ttsplit=pred_full.shape[1])[0]
            # print(rkmap_u_arr_train[0,:10])
            x2y2z2 = sum_numba_axis0(
                (pred_full[:, 1:] - rkmap_u_arr_train[:, :-1]) ** 2.0)
        else:
            x2y2z2 = np.zeros(pred_full[0].size - 1)
            for j in range(1, pred_full[0].size):

                if system == 'lorenz':
                    int_step = math.trunc(tau / 0.01)
                    rkmap_u_arr_train = \
                        numerical_model_wrapped(u0 = np.array([pred_full[0][j - 1] * 7.929788629895004,
                                            pred_full[1][j - 1] * 8.9932616136662,
                                            pred_full[2][j - 1] * 8.575917849311919 + 23.596294463016896]),
                                            int_step = int_step, T=1, tau=tau, system=system, params=params)[0]
                elif system == 'KS':
                    u0 = pred_full[:, j - 1] * (1.1876770355823614)
                    rkmap_u_arr_train = numerical_model_wrapped(tau=tau, T=1, u0=u0, system=system, int_step = 1, params=params)[0]
                elif system == 'KS_d2175':
                    u0 = pred_full[:, j - 1] * (1.2146066380280796)
                    rkmap_u_arr_train = numerical_model_wrapped(tau=tau, T=1, u0=u0, system=system, int_step = 1, params=params)[0]

                x2y2z2[j - 1] = np.sum((pred_full[:, j] - rkmap_u_arr_train[:, 1]) ** 2)
        rms_test = np.sqrt(x2y2z2 / pred_full.shape[0])
        if save_time_rms:
            rms[i] = rms_test
        max_rms[i] = np.max(rms_test)
        mean_rms[i] = np.mean(rms_test)
        if system == 'lorenz':
            means[i] = np.mean(pred_full[0])
            variances[i] = np.var(pred_full[0])
        elif system in ['KS', 'KS_d2175']:
            means[i] = np.mean(pred_full.flatten())
            variances[i] = np.var(pred_full.flatten())
        if mean_rms[i] < 5e-3 and 0.9 < variances[i] and variances[i] < 1.1:
            stable_count += 1
        with objmode(test_toc='double'):
            test_toc = time.perf_counter()
        test_time = test_toc - test_tic

    return stable_count, mean_rms, max_rms, variances, valid_time, rms, preds, wass_dist, pmap_max, pmap_max_wass_dist


def generate_res(run_opts, res_gen, res_itr, rk, noise_stream, noise_scaling=0):
    # Function for generating a reservoir and obtaining matrices used for training the reservoir
    reservoir = Reservoir(run_opts, res_gen, res_itr, rk.u_arr_train.shape[0])
    data_shape = rk.u_arr_train.shape
    res_shape = reservoir.X.shape
    noise_in = gen_noise_driver(run_opts, data_shape, res_shape, noise_scaling, noise_stream)
    get_states(run_opts, reservoir, rk, noise_in, noise_scaling)
    return reservoir, noise_in


def optim_func(run_opts, res_out, res, noise_in, noise, noise_idx, rktest_u_arr_train_nonoise, rktest_u_arr_test, alpha,
               alpha_idx, true_pmap_max, rkTime=400, split=2000, params=np.array([[], []], dtype=np.complex128)):
    # Function for training and testing the performance of a reservoir trained using a particular regularization parameter
    num_eigenvals = 500
    if run_opts.squarenodes:
        res_feature_size = res.rsvr_size * 2
    else:
        res_feature_size = res.rsvr_size
    if run_opts.prior == 'zero':
        idenmat = np.identity(
            res_feature_size + 1 + rktest_u_arr_train_nonoise.shape[0]) * alpha
        prior = np.zeros(res.data_trstates.shape)
    elif run_opts.prior == 'input_pass':
        idenmat = np.identity(
            res_feature_size + 1 + rktest_u_arr_train_nonoise.shape[0]) * alpha
        prior = np.concatenate((np.zeros((rktest_u_arr_train_nonoise.shape[0], 1 + res_feature_size)),
                                np.identity(rktest_u_arr_train_nonoise.shape[0])), axis=1) * alpha

    num_reg_train_times = res.gradient_reg.shape[0]
    res.Wout = np.zeros((num_reg_train_times, rktest_u_arr_train_nonoise.shape[0],
                         rktest_u_arr_train_nonoise.shape[0] + 1 + res_feature_size))
    trainlen_mult = 1.0 / res.Y_train.shape[1]
    for i in range(num_reg_train_times):
        if run_opts.traintype not in ['sylvester', 'sylvester_wD']:
            res.Wout[i] = np.transpose(solve(np.transpose(
                res.states_trstates[i] * trainlen_mult + noise * res.gradient_reg[i] + idenmat),
                np.transpose(res.data_trstates * trainlen_mult + prior)))
        else:
            res.Wout[i] = solve_sylvester(
                res.left_mat[i], res.states_trstates[i] + idenmat, res.data_trstates)
        train_preds = res.Wout[i] @ res.X_train
        train_rms = np.sqrt(np.mean((train_preds - res.Y_train) ** 2.0, axis=0))
        res_out.train_mean_rms_out[:, :, i] = np.mean(train_rms)
        res_out.train_max_rms_out[:, :, i] = np.max(train_rms)
        if run_opts.save_eigenvals and alpha_idx == 0:
            eigenvals_out = np.linalg.eigvalsh(res.gradient_reg[i])
            res_out.grad_eigenvals[:, i] = eigenvals_out[eigenvals_out.size:eigenvals_out.size - num_eigenvals - 1: -1]
        res_out.stable_frac_out[noise_idx, alpha_idx, i], res_out.mean_rms_out[noise_idx, alpha_idx, i], \
        res_out.max_rms_out[noise_idx, alpha_idx, i], res_out.variances_out[noise_idx, alpha_idx, i], \
        res_out.valid_time_out[noise_idx, alpha_idx, i], res_out.rms_out[noise_idx, alpha_idx, i], \
        res_out.pred_out[noise_idx, alpha_idx, i], res_out.wass_dist_out[noise_idx, alpha_idx, i], \
        res_out.pmap_max_out[noise_idx, alpha_idx, i], res_out.pmap_max_wass_dist_out[noise_idx, alpha_idx, i] \
            = test(run_opts, res, i, noise_in, rktest_u_arr_train_nonoise, rktest_u_arr_test,
                   true_pmap_max, rkTime=rkTime, split=split, params=params)


def get_res_results(run_opts, res_itr, res_gen, rk, noise, noise_stream,
                    rktest_u_arr_train_nonoise, rktest_u_arr_test, rkTime_test, split_test, params,
                    train_seed, true_pmap_max):
    # Function for generating, training, and testing the performance of a reservoir given an input set of testing data time series,
    # a set of regularization values, and a set of noise magnitudes
    tic = time.perf_counter()
    print('Starting res %d' % res_itr)
    reservoir, noise_in = generate_res(run_opts, res_gen, res_itr, rk, noise_stream, noise)

    toc = time.perf_counter()
    print('Res states found for itr %d, runtime: %f sec.' % (res_itr, toc - tic))
    num_vt_tests = (rkTime_test - split_test) // run_opts.max_valid_time

    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise
    res_out = ResOutput(run_opts, noise)
    for noise_idx, noise in enumerate(noise_array):
        noise_tic = time.perf_counter()
        min_optim_func = lambda alpha, alpha_idx: optim_func(run_opts, res_out, reservoir, noise_in[noise_idx], noise,
                                                             noise_idx, \
                                                             rktest_u_arr_train_nonoise, rktest_u_arr_test, alpha,
                                                             alpha_idx,
                                                             true_pmap_max, rkTime_test, split_test, params)
        func_vals = np.zeros(run_opts.reg_values.size)
        for j, alpha_value in enumerate(run_opts.reg_values):
            print('Regularization: ', run_opts.reg_values[j])
            if run_opts.debug_mode:
                min_optim_func(alpha_value, j)
            else:
                try:
                    min_optim_func(alpha_value, j)
                except:
                    print('Training unsucessful for alpha:')
                    print(alpha_value)
        noise_toc = time.perf_counter()
        print('Noise test time: %f sec.' % (noise_toc - noise_tic))
    toc = time.perf_counter()
    runtime = toc - tic
    print('Iteration runtime: %f sec.' % runtime)
    return res_out, train_seed, noise_array, res_itr


def find_stability(run_opts, noise, train_seed, train_gen, dnoise_train_gen, res_itr, res_gen, test_stream,
                   dnoise_test_stream, test_idxs, noise_stream, overall_idx):
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
    if run_opts.system == 'lorenz':
        if run_opts.test_time == 0:
            rkTime_test = 4000
        else:
            rkTime_test = run_opts.test_time
        split_test = run_opts.sync_time
    elif run_opts.system in ['KS', 'KS_d2175']:
        if run_opts.test_time == 0:
            rkTime_test = int(16000)
        else:
            rkTime_test = int(run_opts.test_time)
        split_test = int(run_opts.sync_time)

    rktest_u_arr_train_nonoise, rktest_u_arr_test, params = get_test_data(
        run_opts, test_stream, dnoise_test_stream, overall_idx, rkTime=rkTime_test, split=split_test)
    # np.random.seed(train_seed)
    if run_opts.system == 'lorenz':
        ic = train_gen.random(3) * 2 - 1
        rk = NumericalModel(u0 = np.array([ic[0], ic[1], 30 * ic[2]]), tau=run_opts.tau,
                        T=run_opts.train_time + run_opts.discard_time,
                        ttsplit=run_opts.train_time + run_opts.discard_time, system=run_opts.system, params=params)
    elif run_opts.system in ['KS', 'KS_d2175']:
        u0 = 0.6 * (train_gen.random(64) * 2 - 1)
        u0 = u0 - np.mean(u0)
        rk = NumericalModel(tau=run_opts.tau, T=run_opts.train_time + run_opts.discard_time,
                        ttsplit=run_opts.train_time + run_opts.discard_time, u0=u0, system=run_opts.system,
                        params=params)
    if run_opts.pmap:
        true_pmap_max_filename = run_opts.root_folder + \
                                 '%s_tau%0.2f_true_pmap_max.csv' % (run_opts.system, run_opts.tau)
        if os.name == 'nt' and len(true_pmap_max_filename) >= 260:
            true_pmap_max_filename = get_windows_path(true_pmap_max_filename)
        true_pmap_max = np.loadtxt(true_pmap_max_filename, delimiter=',')
        print('Snippet of true poincare map:')
        print(true_pmap_max[:5])
    else:
        true_pmap_max = np.zeros(100)

    res_out, train_seed, noise_array, itr = get_res_results(run_opts, res_itr, res_gen, \
                                                            rk, noise, noise_stream, rktest_u_arr_train_nonoise,
                                                            rktest_u_arr_test, \
                                                            rkTime_test, split_test, params, train_seed, true_pmap_max)

    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise

    res_out.save(run_opts, noise_array, res_itr, train_seed, test_idxs)
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


def start_reservoir_test(argv=None, run_opts=None):
    # Main driver function for obtaining reservoir performance. This function processes input arguments and
    # creates random number generators for the different reservoirs, trainings, noise arrays, and tests.
    # It then calls find_stability in a loop, processes the output from find_stability, and saves the output to a folder.

    if not isinstance(argv, type(None)) and isinstance(run_opts, type(None)):
        run_opts = RunOpts(argv)
    print(run_opts.run_folder_name)
    if run_opts.machine == 'personal':
        if run_opts.ifray:
            ray.init(num_cpus=run_opts.num_cpus)
    elif run_opts.machine == 'deepthought2':
        if run_opts.ifray:
            ray.init(address=os.environ["ip_head"])

    if run_opts.traintype in ['normal', 'normalres1', 'normalres2', 'rmean', 'rmeanres1', 'rmeanres2',
                              'rplusq', 'rplusqres1', 'rplusqres2']:
        run_opts.reg_values = run_opts.reg_values * run_opts.noise_realizations
    if run_opts.reg_train_times is None:
        run_opts.reg_train_times = np.array([run_opts.train_time - run_opts.discard_time])

    ss_res = np.random.SeedSequence(12)
    ss_train = np.random.SeedSequence(34)
    ss_test = np.random.SeedSequence(56)
    ss_noise = np.random.SeedSequence(78)
    ss_dnoise_train = np.random.SeedSequence(910)
    ss_dnoise_test  = np.random.SeedSequence(1112)
    if run_opts.traintype in ['gradient1', 'gradient2',
                              'gradient12'] or 'gradientk' in run_opts.traintype or 'regzerok' in run_opts.traintype:
        res_seeds = ss_res.spawn(run_opts.res_per_test + run_opts.res_start)
        train_seeds = ss_train.spawn(run_opts.num_trains + run_opts.train_start)
        test_seeds = ss_test.spawn(run_opts.num_tests + run_opts.test_start)
        test_idxs = np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)
        res_streams = np.zeros(run_opts.res_per_test * run_opts.num_trains, dtype=object)
        train_streams = np.zeros(run_opts.res_per_test * run_opts.num_trains, dtype=object)
        test_streams = np.zeros((run_opts.res_per_test * run_opts.num_trains, run_opts.num_tests), dtype=object)
        for i in range(run_opts.res_per_test * run_opts.num_trains):
            test_streams[i] = np.array([np.random.default_rng(test_seeds[s]) for \
                                        s in test_idxs], dtype=object)
        noise_streams = np.empty(run_opts.noise_realizations, dtype=object)

        tr, rt = np.meshgrid(np.arange(run_opts.num_trains), np.arange(run_opts.res_per_test))
        tr = tr.flatten() + run_opts.train_start
        rt = rt.flatten() + run_opts.res_start
        for i in range(run_opts.res_per_test * run_opts.num_trains):
            res_streams[i] = np.random.default_rng(res_seeds[rt[i]])
            train_streams[i] = np.random.default_rng(train_seeds[tr[i]])
        dnoise_train_streams = np.zeros(run_opts.res_per_test * run_opts.num_trains, dtype=object)
        dnoise_test_streams = np.zeros((run_opts.res_per_test * run_opts.num_trains, run_opts.num_tests), dtype=object)
        dnoise_train_seeds = ss_dnoise_train.spawn(run_opts.num_trains + run_opts.train_start)
        dnoise_test_seeds = ss_dnoise_test.spawn(run_opts.num_tests + run_opts.test_start)
        for i in range(run_opts.res_per_test * run_opts.num_trains):
            dnoise_test_streams[i] = np.array([np.random.default_rng(dnoise_test_seeds[s]) for \
                                        s in test_idxs], dtype=object)
            dnoise_train_streams[i] = np.random.default_rng(dnoise_train_seeds[tr[i]])
        incomplete_idxs = []
        for i in range(tr.size):
            if not os.path.exists(os.path.join(run_opts.run_folder_name,
                                               'train_max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (
                                                       rt[i], tr[i], run_opts.noise_values_array[-1],
                                                       run_opts.reg_train_times[-1]))):
                incomplete_idxs.append(i)
        print('Total idxs: %d' % tr.size)
        print('Incomplete idxs: %d' % len(incomplete_idxs))

        print('Starting Ray Computation')
        tic = time.perf_counter()
        if run_opts.ifray:
            out_base = ray.get(
                [find_stability_remote.remote(run_opts, run_opts.noise_values_array, tr[i], train_streams[i],
                                              dnoise_train_streams[i], rt[i], res_streams[i], test_streams[i],
                                              dnoise_test_streams[i], test_idxs, noise_streams, i) for i
                 in incomplete_idxs])
        else:
            out_base = [find_stability_serial(run_opts, run_opts.noise_values_array, tr[i], train_streams[i],
                                              dnoise_train_streams[i], rt[i], res_streams[i], test_streams[i],
                                              dnoise_test_streams[i], test_idxs, noise_streams, i) for i
                        in incomplete_idxs]

    else:
        res_seeds = ss_res.spawn(run_opts.res_per_test + run_opts.res_start)
        train_seeds = ss_train.spawn(run_opts.num_trains + run_opts.train_start)
        test_seeds = ss_test.spawn(run_opts.num_tests + run_opts.test_start)
        test_idxs = np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)
        noise_seeds = ss_noise.spawn(run_opts.noise_realizations)
        res_streams = np.zeros(run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size,
                               dtype=object)
        train_streams = np.zeros(run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size,
                                 dtype=object)
        test_streams = np.zeros(
            (run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size, run_opts.num_tests),
            dtype=object)
        noise_streams = np.zeros((run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size,
                                  run_opts.noise_realizations), dtype=object)

        tnr, ntr, rtn = np.meshgrid(np.arange(run_opts.num_trains), run_opts.noise_values_array,
                                    np.arange(run_opts.res_per_test))
        tnr = tnr.flatten() + run_opts.res_start
        ntr = ntr.flatten()
        rtn = rtn.flatten() + run_opts.train_start

        for i in range(run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size):
            # print(res_seeds[rtn[i]])
            res_streams[i] = np.random.default_rng(res_seeds[rtn[i]])
            train_streams[i] = np.random.default_rng(train_seeds[tnr[i]])
            test_streams[i] = np.array([np.random.default_rng(test_seeds[j]) for \
                                        j in test_idxs])
            noise_streams[i] = np.array([np.random.default_rng(j) for j in noise_seeds])
        dnoise_train_streams = np.zeros(run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size,
                                        dtype=object)
        dnoise_test_streams = np.zeros(
            (run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size, run_opts.num_tests),
            dtype=object)
        dnoise_train_seeds = ss_dnoise_train.spawn(run_opts.num_trains + run_opts.train_start)
        dnoise_test_seeds = ss_dnoise_test.spawn(run_opts.num_tests + run_opts.test_start)
        for i in range(run_opts.num_trains * run_opts.res_per_test * run_opts.noise_values_array.size):
            dnoise_test_streams[i] = np.array([np.random.default_rng(dnoise_test_seeds[j]) for \
                                        j in test_idxs])
            dnoise_train_streams[i] = np.random.default_rng(dnoise_train_seeds[tnr[i]])
        incomplete_idxs = []
        for i in range(tnr.size):
            if not os.path.exists(os.path.join(run_opts.run_folder_name,
                                               'train_max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (
                                                       rtn[i], tnr[i], ntr[i], run_opts.reg_train_times[-1]))):
                incomplete_idxs.append(i)

        print('Starting Ray Computation')
        tic = time.perf_counter()
        if run_opts.ifray:
            out_base = ray.get([find_stability_remote.remote(run_opts, ntr[i], tnr[i], train_streams[i],
                                                             dnoise_train_streams[i], rtn[i], res_streams[i],
                                                             test_streams[i], dnoise_test_streams[i], test_idxs,
                                                             noise_streams[i], i) for i in incomplete_idxs])
        else:
            out_base = [find_stability_serial(run_opts, ntr[i], tnr[i], train_streams[i], dnoise_train_streams[i],
                                              rtn[i], res_streams[i], test_streams[i], dnoise_test_streams[i],
                                              test_idxs, noise_streams[i], i) for i in incomplete_idxs]
    if len(incomplete_idxs) != 0:
        toc = time.perf_counter()
        runtime = toc - tic
        print('Runtime over all cores: %f sec.' % (runtime))
        print('Ray finished.')
        print('Results Saved')
    else:
        print('No incomplete runs were found. Ending job.')

    if run_opts.ifray:
        ray.shutdown()


def main(argv):
    start_reservoir_test(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
