#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 4:00:00
#Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=1
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mem-per-cpu=6144
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
"""
Created on Wed Jun  3 12:57:49 2020

@author: josephharvey
"""
import sys, getopt, os
sys.path.append('/lustre/awikner1/res-noise-stabilization/')

from datetime import datetime
# from lorenzrungekutta_numba import fx
# from lorenzrungekutta_numba import fy
# from lorenzrungekutta_numba import fz
import numpy as np
#from sklearn.linear_model import Ridge
from scipy.linalg import solve, solve_sylvester
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from csr import CSR
from numba import jit, njit
from numba.experimental import jitclass
from numba.types import int32, int64, double
import time
import cProfile, pstats

import pkg_resources, os
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
isray = [('ray==' in elem) for elem in installed_packages_list]
if (True in isray):
    print('Ray installed')
else:
    os.system('pip install -r -U ray')

import ray
from lorenzrungekutta_numba import *
from ks_etdrk4 import *
from csc_mult import *

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

@njit
def str_to_int(s):
    final_index, result = len(s) - 1, 0
    for i,v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result

@jit(nopython = True, fastmath = True)
def mean_numba_axis1(mat):

    res = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        res[i] = np.mean(mat[i])

    return res

@jit(nopython = True, fastmath = True)
def sum_numba_axis0(mat):

    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.sum(mat[:,i])
    return res
"""
def create_csc_matrix(data, indices, indptr, shape):
    return csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))

@jit(nopython = True)
def matrix_dot_left_T(data, indices, indptr, shape, mat):
    with objmode(out = 'double[:,:]'):
        csc = create_csc_matrix(data, indices, indptr, shape)
        out = csc.T.dot(mat).T
    return out

@jit(nopython = True, fastmath = True)
def mult_vec(data, indices, indptr, shape, mat):
    out = np.zeros(shape[0])
    for i in range(mat.size):
        for k in range(indptr[i], indptr[i+1]):
            out[indices[k]] += data[k] * mat[i]
    return out

@jit(nopython = True, fastmath = True)
def construct_jac_mat_csc(Win, data_in, indices_in, indptr_in, shape_in, rsvr_size, n):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int64[:]'):
        W_conv = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0], shape_in[1])).toarray()
        mat    = csc_matrix(np.concatenate((Win[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n))), axis = 1))
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape))
    return data, indices, indptr, shape
"""

class Reservoir:
    def __init__(self, rk, input_size, rsvr_size = 300, spectral_radius = 0.6, input_weight = 1, leakage = 1.0, win_type = 'full', bias_type = 'old', res_seed = 1):
        self.rsvr_size = rsvr_size
        """
        print('Spectral Radius: %0.2f' % spectral_radius)
        print('Input Weight: %0.2f' % input_weight)
        print('Leakage: %0.3f' % leakage)
        print('Win Type: %s' % win_type)
        print('Bias type: %s' % bias_type)
        """

        #get spectral radius < 1
        #gets row density = 0.03333
        avg_degree = 10
        density = avg_degree/rsvr_size
        np.random.seed(res_seed)
        # unnormalized_W = sparse.random(rsvr_size, rsvr_size, density).todense()*2-1

        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 10/rsvr_size:
                    unnormalized_W[i][j] = 0


        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False, maxiter = 10**5)

        # self.W = sparse.csr_matrix(spectral_radius/np.abs(max_eig)*unnormalized_W)
        W_sp    = csc_matrix(np.ascontiguousarray(spectral_radius/np.abs(max_eig[0])*unnormalized_W))
        self.W_data, self.W_indices, self.W_indptr, self.W_shape  = \
                (W_sp.data, W_sp.indices, W_sp.indptr, np.array(list(W_sp.shape)))

        if win_type == 'full':
            input_vars = np.arange(input_size)
        elif win_type == 'x':
            input_vars = np.array([0])
        if bias_type == 'old':
            const_frac = 0.15
            const_conn = int(rsvr_size*const_frac)
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:const_conn, 0] = (np.random.rand(Win[:const_conn, 0].size)*2 - 1)*input_weight
            q = int((rsvr_size-const_conn)//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[const_conn+q*i:const_conn+q*(i+1),var+1] = (np.random.rand(q)*2-1)*input_weight
        elif bias_type == 'new_random':
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:,0] = (np.random.rand(rsvr_size)*2-1)*input_weight
            q = int(rsvr_size//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[q*i:q*(i+1),var+1] = (np.random.rand(q)*2-1)*input_weight
        elif bias_type == 'new_const':
            Win = np.zeros((rsvr_size, input_size+1))
            Win[:,0] = input_weight
            q = int(rsvr_size//input_vars.size)
            for i, var in enumerate(input_vars):
                Win[q*i:q*(i+1),var+1] = (np.random.rand(q)*2-1)*input_weight


        # self.Win = sparse.csr_matrix(Win)
        self.Win = np.ascontiguousarray(Win)

        self.X = (np.random.rand(rsvr_size, rk.train_length+2)*2 - 1)
        self.Wout = np.array([])
        self.leakage = leakage

class RungeKutta:
    def __init__(self, x0 = 2,y0 = 2,z0 = 23, h = 0.01, tau = 0.1, T = 300, ttsplit = 5000, u0 = 0, system = 'lorenz', params = np.array([[],[]], dtype = np.complex128)):
        if system == 'lorenz':
            int_step = int(tau/h)
            u_arr = np.ascontiguousarray(rungekutta(x0,y0,z0,h,T,tau)[:, ::int_step])
            self.input_size = 3

            u_arr[0] = (u_arr[0] - 0)/7.929788629895004
            u_arr[1] = (u_arr[1] - 0)/8.9932616136662
            u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
            self.params = params

        elif system == 'KS':
            u_arr, self.params = kursiv_predict(u0, tau = tau, T = T, params = params)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
        else:
            raise ValueError



        self.train_length = ttsplit
        self.u_arr_train = u_arr[:, :ttsplit+1]
        #size 5001

        #noisy training array
        #switch to gaussian

        #plt.plot(self.u_arr_train_noise[0, :500])

        #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]
        #size 1001

@jit(nopython = True, fastmath = True)
def RungeKuttawrapped(x0 = 2,y0 = 2,z0 = 23, h = 0.01, tau = 0.1, T = 300, ttsplit = 5000, u0 = 0, system = 'lorenz', params = np.array([[],[]], dtype = np.complex128)):
    if system == 'lorenz':
        int_step = int(tau/h)
        u_arr = np.ascontiguousarray(rungekutta(x0,y0,z0,h,T,tau)[:, ::int_step])
        # self.train_length = ttsplit
        # self.noise_scaling = noise_scaling

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict(u0, tau = tau, T = T, params = params)
        u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    #size 5001

    #noisy training array
    #switch to gaussian

    #plt.plot(self.u_arr_train_noise[0, :500])

    #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params

@jit(nopython = True, fastmath = True)
def RungeKuttawrapped_pred(h = 0.01, tau = 0.1, T = 300, ttsplit = 5000, u0_array = np.array([[],[]], dtype = np.complex128), system = 'lorenz', params = np.array([[],[]], dtype = np.complex128)):
    if system == 'lorenz':
        int_step = int(tau/h)
        u_arr = np.ascontiguousarray(rungekutta_pred(u0_array,h,tau,int_step))
        # self.train_length = ttsplit
        # self.noise_scaling = noise_scaling

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        new_params = params
    elif system == 'KS':
        u_arr, new_params = kursiv_predict_pred(u0_array, tau = tau, T = T, params = params)
        u_arr = np.ascontiguousarray(u_arr)/(1.1876770355823614)
    else:
        raise ValueError

    u_arr_train = u_arr[:, :ttsplit+1]
    #size 5001

    #noisy training array
    #switch to gaussian

    #plt.plot(self.u_arr_train_noise[0, :500])

    #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_test, ttsplit, new_params

def getX(res, rk,x0 = 1,y0 = 1,z0 = 1):
    u_training = rk.u_arr_train
    res.X = getXwrapped(np.ascontiguousarray(u_training), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage)

    return res.X
#takes a reservoir object res along with initial conditions


@jit(nopython = True, fastmath = True)
def getXwrapped(u_training, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype = 'none', noise_scaling = 0, noise_realization = 0, traintype = 'normal'):

    #loops through every timestep
    if noisetype in ['gaussian', 'perturbation']:
        if traintype in ['normal','rmean','rplusq']:
            noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0-leakage)*res_X[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i]+noise[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:,i]))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1','rmeanres1','rplusqres1']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0-leakage)*res_X[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:,i]+noise[:,i]))
            u_training_wnoise = u_training
        elif traintype in ['normalres2','rmeanres2','rplusqres2']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0-leakage)*res_X[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:,i])+noise[:,i])
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif noisetype not in ['gaussian_onestep','perturbation_onestep'] and \
            ('gaussian' in noisetype and 'step' in noisetype):
        noise_steps = str_to_int(noisetype.replace('gaussian','').replace('perturbation','').replace('step',''))
        res_X_nonoise = np.copy(res_X)
        for i in range(0, u_training[0].size):
            res_X_nonoise[:,i+1] = (1.0-leakage)*res_X_nonoise[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:,i]))
        if traintype in ['normal','rmean','rplusq']:
            noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:,i]
                temp_x = (1.0-leakage)*res_X_nonoise[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i]+noise[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i+k]+noise[:,i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                res_X[:,i+noise_steps] = temp_x
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1','rmeanres1','rplusqres1']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:,i]
                temp_x = (1.0-leakage)*res_X_nonoise[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:,i]))
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:,i+k]))
                res_X[:,i+noise_steps] = temp_x
            u_training_wnoise = u_training
        elif traintype in ['normalres2','rmeanres2','rplusqres2']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            for i in range(0, u_training[0].size-noise_steps):
                temp_x = res_X_nonoise[:,i]
                temp_x = (1.0-leakage)*res_X_nonoise[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:,i])
                for k in range(1, noise_steps):
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:,i+k])
                res_X[:,i+noise_steps] = temp_x
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif noisetype in ['gaussian_onestep','perturbation_onestep']:
        if traintype in ['normal','rmean','rplusq']:
            noise = gen_noise(u_training.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            temp_x = res_X[:,0]
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i]+noise[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                temp_x = np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training+noise
        elif traintype in ['normalres1','rmeanres1','rplusqres1']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            temp_x = res_X[:,0]
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x+noise[:,i]))
                temp_x = np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        elif traintype in ['normalres2','rmeanres2','rplusqres2']:
            noise = gen_noise(res_X.shape[0], u_training.shape[1], noisetype, noise_scaling, noise_realization)
            temp_x = res_X[:,0]
            for i in range(0, u_training[0].size):
                res_X[:,i+1] = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x)+noise[:,i])
                temp_x = np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            u_training_wnoise = u_training
        else:
            raise ValueError
        return res_X, u_training_wnoise
    elif 'gaussian' in noisetype:
        noise_steps     = str_to_int(noisetype.replace('gaussian',''))

        noise = gen_noise(u_training.shape[0], u_training.shape[1], str(noisetype), noise_scaling, noise_realization)
        res_X_nonoise = np.copy(res_X)
        for i in range(0, u_training[0].size):
            res_X_nonoise[:,i+1] = (1.0-leakage)*res_X_nonoise[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X_nonoise[:,i]))
        for i in range(0, u_training[0].size-noise_steps):
            temp_x = res_X_nonoise[:,i]
            for k in range(noise_steps):
                if k == 0:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i+k]+noise[:,i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
                else:
                    temp_x = (1.0 - leakage)*temp_x + leakage*np.tanh(Win @ np.append(1., u_training[:,i+k])+mult_vec(W_data, W_indices, W_indptr, W_shape, temp_x))
            res_X[:,i+noise_steps] = temp_x
        u_training_wnoise = u_training+noise
        return res_X, u_training_wnoise


    elif traintype in ['sylvester_wD'] or 'gradient' in traintype:
        res_D = np.zeros((res_X.shape[0], u_training.shape[1]+1))
        for i in range(0, u_training[0].size):
            res_internal = Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:,i])
            res_X[:,i+1] = (1.0 - leakage)*res_X[:,i] + leakage*np.tanh(res_internal)
            res_D[:,i+1] = leakage/(np.power(np.cosh(res_internal),2.0))
        return res_X, res_D
    else:
        for i in range(0, u_training[0].size):
            res_X[:,i+1] = (1.0 - leakage)*res_X[:,i] + leakage*np.tanh(Win @ np.append(1., u_training[:,i])+mult_vec(W_data, W_indices, W_indptr, W_shape, res_X[:,i]))
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

@jit(nopython = True, fastmath = True)
def gen_noise(noise_size, noise_length, noisetype, noise_scaling, noise_realization):
    if 'gaussian' in noisetype:
        np.random.seed(noise_realization+9)
        noise = np.random.randn(noise_size, noise_length)*noise_scaling
    if noisetype in ['perturbation', 'perturbation_onestep']:
        noise = np.zeros((noise_size, noise_length))
        if noise_realization < noise_size:
            noise[noise_realization] = np.ones(noise_length)*noise_scaling
        elif noise_realization < 2*noise_length:
            noise[noise_realization-noise_size] = -np.ones(noise_length)*noise_scaling
        else:
            raise ValueError

    return noise

def get_states(res, rk, noisetype = 'none', noise_scaling = 0, noise_realizations = 1,\
        traintype = 'normal', skip = 150):
    if traintype == 'getD':
        Dn = getD(np.ascontiguousarray(rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage, \
             skip, noisetype, noise_scaling, noise_realizations, traintype)
        return Dn
    elif traintype not in ['sylvester','sylvester_wD']:
        res.data_trstates, res.states_trstates, res.gradient_reg = get_states_wrapped(\
            np.ascontiguousarray(rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,\
            skip, noisetype, noise_scaling, noise_realizations, traintype)
    else:
        res.data_trstates, res.states_trstates, res.left_mat  = get_states_wrapped(\
            np.ascontiguousarray(rk.u_arr_train), res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.leakage,\
            skip, noisetype, noise_scaling, noise_realizations, traintype)

@jit(nopython = True, fastmath = True)
def get_states_wrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, skip, noisetype = 'none',\
        noise_scaling = 0, noise_realizations = 1, traintype = 'normal', q = 0):
    res_X = np.ascontiguousarray(res_X)
    u_arr_train = np.ascontiguousarray(u_arr_train)
    n,d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
    Y_train = np.ascontiguousarray(u_arr_train[:, skip+1:])
    if traintype in ['normal', 'normalres1','normalres2']:
        data_trstates = np.zeros((n,rsvr_size+1+n), dtype = np.float64)
        states_trstates = np.zeros((n+rsvr_size+1,n+rsvr_size+1), dtype = np.float64)
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, \
                noise_scaling, i, traintype)
            X = X[:, skip+1:(res_d - 1)]
            X_train = np.ascontiguousarray(np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis = 0))
            data_trstates += Y_train @ X_train.T
            states_trstates += X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rmean','rmeanres1','rmeanres2']:
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, \
                noise_scaling, i, traintype)
            X = X[:, skip+1:(res_d - 1)]
            X_train = np.ascontiguousarray(np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis = 0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
        data_trstates = Y_train @ X_train_mean.T
        states_trstates = X_train_mean @ X_train_mean.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype in ['rplusq','rplusqres1','rplusqres2']:
        X_0, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage)
        X_0 = np.ascontiguousarray(X_0[:, skip+1:(res_X[0].size - 1)])
        X_train_0 = np.ascontiguousarray(np.concatenate((np.ones((1, d-(skip+1))), X_0, u_arr_train[:, skip:-1]), axis = 0))
        X_all = np.zeros((X_train_0.shape[0], X_train_0.shape[1], noise_realizations))
        for i in range(noise_realizations):
            X, u_arr_train_noise = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, \
                noise_scaling, i, traintype)
            X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
            X_train = np.ascontiguousarray(np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis = 0))
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations
            X_all[:,:,i] = X_train
        Y_fit = Y_train
        X_fit = X_all[:,:,0] - X_train_mean + X_train_0
        for i in range(1,noise_realizations):
            Y_fit = np.append(Y_fit, Y_train, axis = 1)
            X_fit = np.append(X_fit, X_all[:,:,i] - X_train_mean + X_train_0, axis = 1)
        data_trstates = Y_fit @ X_fit.T
        states_trstates = X_fit @ X_fit.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0)
        gradient_reg = np.zeros((rsvr_size+n, rsvr_size+n))
        for i in range(X.shape[1]):
            D_n = np.concatenate((matrix_diag_mult(D[:,i], Win[:,1:]), np.identity(n)), axis = 0)
            gradient_reg += D_n @ D_n.T
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        states_trstates[1:,1:] = states_trstates[1:,1:] + noise_scaling**2/noise_realizations*gradient_reg
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient12':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        for i in range(X.shape[1]):
            D_n  = np.concatenate((matrix_diag_mult(D[:,i], Win[:,1:]), np.identity(n)), axis = 0)
            gradient_reg[1:,1:] += D_n @ D_n.T
        for i in range(1, X.shape[1]):
            D_n2 = np.concatenate((np.zeros((1,n)), matrix_diag_mult(D[:,i-1], Win[:,1:]), np.identity(n)), axis = 0)
            E_n  = np.concatenate((np.zeros((1, rsvr_size+n+1)),matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, np.diag(D[:,i])) + \
                np.concatenate((np.zeros((1,rsvr_size)), (1-leakage)*np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis = 1),\
                np.zeros((n, rsvr_size+n+1))), axis = 0)
            E_nD_n2   = E_n @ D_n2
            gradient_reg += E_nD_n2 @ E_nD_n2.T
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        states_trstates = states_trstates + noise_scaling**2/noise_realizations*gradient_reg
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'gradient2':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        for i in range(1, X.shape[1]):
            D_n2 = np.concatenate((np.zeros((1,n)), matrix_diag_mult(D[:,i-1], Win[:,1:]), np.identity(n)), axis = 0)
            E_n  = np.concatenate((np.zeros((1, rsvr_size+n+1)),matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, np.diag(D[:,i])) + \
                np.concatenate((np.zeros((1,rsvr_size)), (1-leakage)*np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis = 1),\
                np.zeros((n, rsvr_size+n+1))), axis = 0)
            E_nD_n2   = E_n @ D_n2
            gradient_reg += E_nD_n2 @ E_nD_n2.T
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        states_trstates = states_trstates + noise_scaling**2/noise_realizations*gradient_reg
        return [data_trstates, states_trstates, gradient_reg]
    elif 'gradientk' in traintype:
        k = str_to_int(traintype.replace('gradientk',''))
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0)
        gradient_reg = np.zeros((rsvr_size+n+1, rsvr_size+n+1))
        D_n = np.zeros((rsvr_size+n+1, n, k))
        E_n = np.zeros((rsvr_size+n+1, rsvr_size+n+1, k-1))
        reg_components = np.zeros((rsvr_size+n+1, n, k))
        W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc(Win, W_data, W_indices, W_indptr, W_shape, rsvr_size, n)
        leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage)*np.identity(rsvr_size), np.zeros((rsvr_size, n))), axis = 1)
        """
        with objmode(tic = 'double'):
            tic = time.perf_counter()
        """
        for i in range(k):
            D_n[1:rsvr_size+1,:,i] = matrix_diag_mult(D[:,i], Win[:,1:])
            D_n[rsvr_size+1:,:,i] = np.identity(n)
        for i in range(1,k):
            E_n[1:rsvr_size+1,:,i-1]  = matrix_diag_sparse_mult_add(D[:,i], W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat)
        reg_components[:,:,k-1] = D_n[:,:,-1]
        for i in range(k-1):
            reg_components[:,:,i] = D_n[:,:,i]
            for j in range(i,k-1):
                reg_components[:,:,i] = matrix_sparse_mult(E_n[:,:,j], reg_components[:,:,i])

        for i in range(k, X.shape[1]):
            for j in range(k):
                gradient_reg += reg_components[:,:,j] @ reg_components[:,:,j].T
            reg_components[1:rsvr_size+1,:,k-1] = matrix_diag_mult(D[:,i], Win[:,1:])
            reg_components[1+rsvr_size:,:,k-1] = np.identity(n)
            E_n[1:rsvr_size+1,:,k-2] = matrix_diag_sparse_mult_add(D[:,i], W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat)
            for j in range(k-1):
                reg_components[:,:,j] = matrix_sparse_mult(E_n[:,:,k-2], reg_components[:,:,j])
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
            gradient_reg += reg_components[:,:,j] @ reg_components[:,:,j].T

        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, gradient_reg]
    elif traintype == 'sylvester':
        X, p = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = np.ascontiguousarray(X[:, skip+1:(res_d - 1)])
        X_train = np.ascontiguousarray(np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0))
        data_trstates   = Y_train @ X_train.T
        data_trstates[:,1:rsvr_size+1] += noise_scaling**2/noise_realizations * matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, Win[:,1:])
        states_trstates = X_train @ X_train.T
        left_mat        = -noise_scaling**2/noise_realizations *(Win[:,1:].T @ Win[:,1:])
        return [data_trstates, states_trstates, left_mat]
    elif traintype == 'sylvester_wD':
        X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, noisetype, noise_scaling, 1, traintype)
        X = X[:, skip+1:(res_d - 1)]
        D = D[:, skip+1:(res_d - 1)]
        X_train = np.concatenate((np.ones((1, d-(skip+1))), X, u_arr_train[:, skip:-1]), axis = 0)
        Dmean             = mean_numba_axis1(D)
        temp_mat = np.diag(Dmean) @ Win[:,1:]
        target_correction = matrix_dot_left_T(W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, temp_mat)
        left_mat          = temp_mat.T @ temp_mat
        target_correction = noise_scaling**2/noise_realizations * target_correction
        left_mat = -noise_scaling**2/noise_realizations * left_mat
        data_trstates   = Y_train @ X_train.T
        data_trstates[:,1:rsvr_size+1] += target_correction
        states_trstates = X_train @ X_train.T
        return [data_trstates, states_trstates, left_mat]
    else:
        data_trstates = np.zeros((n,rsvr_size+1+n), dtype = np.float64)
        states_trstates = np.zeros((n+rsvr_size+1,n+rsvr_size+1), dtype = np.float64)
        return [data_trstates, states_trstates, gradient_reg]

@jit(nopython = True, fastmath = True)
def getD(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, skip, noisetype = 'none',\
         noise_scaling = 0, noise_realizations = 1, traintype = 'normal'):
    n,d = u_arr_train.shape
    rsvr_size, res_d = res_X.shape
    Y_train = u_arr_train[:, skip+1:]
    X, D = getXwrapped(u_arr_train, res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage, 'none',0, 0, 'gradient')
    return [ D[:, skip+1:(res_d - 1)] ]

#CONCATENATE ALL THE DATA BEFORE RUNNING REGRESSION

def predict(res, u0,  steps = 1000):
    Y = predictwrapped(res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.Wout, res.leakage, u0, steps)
    return Y

@jit(nopython = True, fastmath = True)
def predictwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape, Wout, leakage, u0, steps):
    Y = np.empty((Win.shape[1]-1, steps + 1))
    X = np.empty((res_X.shape[0], steps + 1))

    Y[:,0] = u0
    X[:,0] = res_X[:,-2]


    for i in range(0, steps):
        # y_in = Y[:,i].reshape(3,1)
        # x_prev = X[:,i].reshape(res.rsvr_size,1)
        X[:,i+1] = (1-leakage)*X[:,i] + leakage*np.tanh(Win @ np.append(1., Y[:,i]) + mult_vec(W_data, W_indices, W_indptr, W_shape, X[:,i]))
        #X = np.concatenate((X, x_current), axis = 1)
        Y[:,i+1] = Wout @ np.concatenate((np.array([1.]), X[:,i+1], Y[:,i]))
        #y_out = np.matmul(res.Wout, x_current)
        #Y[:,i+1] = y_out


    return Y

@jit(nopython = True, fastmath = True)
def get_test_data(tau, num_tests, rkTime, split, system = 'lorenz'):
    np.random.seed(0)
    if system == 'lorenz':
        ic = np.random.rand(3)*2-1
        u0 = np.zeros(64)
    elif system == 'KS':
        ic = np.zeros(3)
        u0 = (np.random.rand(64)*2-1)*0.6
    u_arr_train_nonoise, u_arr_test, p, params = RungeKuttawrapped(x0 = ic[0], \
         y0 = ic[1], z0 = 30*ic[2], tau=tau, T = rkTime, ttsplit = split, u0 = u0, system = system)
    rktest_u_arr_train_nonoise = np.zeros((u_arr_train_nonoise.shape[0], u_arr_train_nonoise.shape[1], num_tests))
    rktest_u_arr_test = np.zeros((u_arr_test.shape[0], u_arr_test.shape[1], num_tests))
    rktest_u_arr_train_nonoise[:,:,0] = u_arr_train_nonoise
    rktest_u_arr_test[:,:,0] = u_arr_test
    for i in range(1,num_tests):
        np.random.seed(i)
        if system == 'lorenz':
            ic = np.random.rand(3)*2-1
            u0 = np.zeros(64)
        elif system == 'KS':
            ic = np.zeros(3)
            u0 = (np.random.rand(64)*2-1)*0.6
        rktest_u_arr_train_nonoise[:,:,i], rktest_u_arr_test[:,:,i], p, params = RungeKuttawrapped(x0 = ic[0], \
             y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split, u0 = u0, system = system, params = params)

    return rktest_u_arr_train_nonoise, rktest_u_arr_test, params

def test(res, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests = 100, rkTime = 1000, split = 3000, showMapError = False, showTrajectories = False, showHist = False, system = 'lorenz', params = np.array([[],[]], dtype = np.complex128)):
    # tic = time.perf_counter()
    stable_count, mean_sum_squared, variances, valid_time, preds = testwrapped(res.X, res.Win, res.W_data, res.W_indices, res.W_indptr, res.W_shape, res.Wout, res.leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, rkTime, split, showMapError, showTrajectories, showHist, system, params = params)
    # toc = time.perf_counter()
    # runtime = toc - tic
    # print("Test " + str(i) + " valid time: " + str(j))

    return stable_count/num_tests, mean_sum_squared, variances, valid_time, preds

@jit(nopython = True, fastmath = True)
def testwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape, Wout, leakage, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, rkTime, split, showMapError = True,   showTrajectories = True, showHist = True, system = 'lorenz', tau = 0.1, params = np.array([[],[]], dtype = np.complex128)):
    stable_count = 0
    valid_time = np.zeros(num_tests)
    max_sum_square = np.zeros(num_tests)
    mean_sum_square = np.zeros(num_tests)
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    preds = np.zeros((num_tests, rktest_u_arr_test.shape[0], (rkTime-split)+1))

    #print(num_tests)
    for i in range(num_tests):
        with objmode(test_tic = 'double'):
            test_tic = time.perf_counter()
        """
        np.random.seed(i)
        ic = np.random.rand(3)*2-1
        u_arr_train, u_arr_train_nonoise, u_arr_test, train_length, noise_scaling = \
            RungeKuttawrapped(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)
        """
        res_X = (np.zeros((res_X.shape[0], split+2))*2 - 1)
        #print('Win')
        #print(Win[:3,:3])
        #print('A')
        #print(W[:3,:3])
        #print('Wout')
        #print(Wout[:3,:3])

        #sets res.X
        res_X, p = getXwrapped(np.ascontiguousarray(rktest_u_arr_train_nonoise[:,:,i]), res_X, Win, W_data, W_indices, W_indptr, W_shape, leakage)
        pred = predictwrapped(res_X, Win, W_data, W_indices, W_indptr, W_shape, Wout, leakage, u0 = rktest_u_arr_test[:,0,i], steps = (rkTime-split))
        #print(pred[0,:10])
        preds[i] = pred
        error = np.zeros(pred[0].size)
        #print(pred.size)

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
            error[j] = np.sqrt(np.mean((pred[:,j]-rktest_u_arr_test[:,j,i])**2.0))

            if error[j] < vt_cutoff and check_vt:
                valid_time[i] = j
            else:
                check_vt = False
        if array_compute:
            if system == 'lorenz':
                rkmap_u_arr_train = RungeKuttawrapped_pred(u0_array = np.stack((pred[0]*7.929788629895004,\
                    pred[1]*8.9932616136662, pred[2]*8.575917849311919+23.596294463016896)),\
                    h=0.01, system = system, params = params, tau = tau, ttsplit = pred.shape[1])[0]
            elif system == 'KS':
                u0 = pred*1.1876770355823614
                rkmap_u_arr_train = RungeKuttawrapped_pred(u0_array = u0, h=tau, T=1, system = system, params = params, ttsplit = pred.shape[1])[0]
            #print(rkmap_u_arr_train[0,:10])
            x2y2z2 = sum_numba_axis0((pred[:,1:]-rkmap_u_arr_train[:,:-1])**2.0)
        else:
            x2y2z2 = np.zeros(pred[0].size-1)
            for j in range(1, pred[0].size):

                if system == 'lorenz':
                    rkmap_u_arr_train = RungeKuttawrapped(pred[0][j-1]*7.929788629895004, pred[1][j-1]*8.9932616136662, pred[2][j-1]*8.575917849311919+23.596294463016896, h=0.01, T=1, tau = tau, system = system, params = params)[0]
                elif system == 'KS':
                    u0 = pred[:,j-1]*(1.1876770355823614)
                    rkmap_u_arr_train = RungeKuttawrapped(0, 0, 0, h=tau, T=1, u0 = u0, system = system, params = params)[0]
                #if j <= 10:
                #print(rkmap_u_arr_train[0,1])

                x2y2z2[j-1] = np.sum((pred[:,j]-rkmap_u_arr_train[:,1])**2)
        x2y2z2 = x2y2z2/np.sqrt(2.0)
        #print("Mean: " + str(np.mean(pred[0])))
        #print("Variance: " + str(np.var(pred[0])))
        """
        if showHist:
            plt.figure()
            plt.hist(pred[0], bins = 11, label = "Predictions", alpha = 0.75)
            plt.hist(rktest_u_arr_test[0,:,i], bins = 11, label = "Truth", alpha = 0.75)
            plt.legend(loc="upper right")

        if showMapError:
            #plt.figure()
            #plt.plot(vector_field, label = "Vector Field Stability Metric")
            #plt.legend(loc="upper right")

            plt.figure()
            plt.plot(x2y2z2, label = "x + y + z square error")
            plt.legend(loc="upper right")

        if showTrajectories:
            plt.figure()
            plt.plot(pred[0], label = "Predictions")
            plt.plot(rktest_u_arr_test[0,:,i], label = "Truth")
            plt.ylim(-3,3)
            plt.legend(loc="upper right")

        print("Variance of lorenz data x dim: " + str(np.var(rktest_u_arr_test[0,:,i])))
        print("Variance of predictions: " + str(np.var(pred[0])))
        print("Max of total square error: " + str(np.max(x2y2z2)))
        print("Mean of total error: " + str(np.mean(x2y2z2)))
        print("Wasserstein distance: " + str(wasserstein_distance(pred[0], rktest_u_arr_test[0,:,i])))
        print()
        """

        max_sum_square[i] = np.max(x2y2z2)
        mean_sum_square[i] = np.mean(x2y2z2)
        #print(mean_sum_square)
        if system ==  'lorenz':
            means[i] = np.mean(pred[0])
            variances[i] = np.var(pred[0])
        elif system == 'KS':
            means[i] = np.mean(pred.flatten())
            variances[i] = np.var(pred.flatten())

        #print('Map error: ', mean_sum_square[i])
        #print('Variance: ', variances[i])
        #print('True Variance: ', np.var(rktest_u_arr_test))
        if mean_sum_square[i] < 0.01 and 0.95 < variances[i] and variances[i] < 1.1:
            stable_count += 1
            # print("stable")
            # print()
        # else:
            # print("unstable")
            # print()
        with objmode(test_toc = 'double'):
            test_toc = time.perf_counter()
        test_time = test_toc - test_tic
        #print(test_time)

    """
    if showMapError or showTrajectories or showHist:
        plt.show()

    #print("Variance of total square error: " + str(np.var(x2y2z2)))

    print("Avg. max sum square: " + str(np.mean(max_sum_square)))
    print("Avg. mean sum square: " + str(np.mean(mean_sum_square)))
    print("Avg. of x dim: " + str(np.mean(means)))
    print("Var. of x dim: " + str(np.mean(variances)))
    print()
    """


    return stable_count, mean_sum_square, variances, valid_time, preds

def generate_res(itr, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype = 'none', noise_scaling = 0, noise_realizations = 1, traintype = 'normal', skip = 150):


    reservoir = Reservoir(rk, rk.u_arr_train.shape[0], rsvr_size = res_size, \
                spectral_radius = rho, input_weight = sigma, leakage = leakage, win_type=win_type, bias_type = bias_type, res_seed = itr)
    #print('Train Data shape: (%d, %d)' % (rk.u_arr_train.shape[0], rk.u_arr_train.shape[1]))
    #print(rk.u_arr_train[-3:,-3:])
    get_states(reservoir, rk, noisetype, noise_scaling, noise_realizations, traintype, skip)
    return reservoir

def optim_func(res, noise, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha, rkTime = 400, split = 2000, traintype = 'normal', system = 'lorenz', params = np.array([[],[]], dtype = np.complex128)):

    #try:
    idenmat = np.identity(res.rsvr_size+1+rktest_u_arr_train_nonoise.shape[0])*alpha
    if traintype not in ['sylvester','sylvester_wD']:
        res.Wout = np.transpose(solve(np.transpose(res.states_trstates + noise**2.0*res.gradient_reg+idenmat),np.transpose(res.data_trstates)))
    else:
        res.Wout = solve_sylvester(res.left_mat, res.states_trstates+idenmat, res.data_trstates)
    out =  test(res, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests = num_tests, rkTime = rkTime, split =split,  \
        showMapError = True, showTrajectories = True, showHist = True, system = system, params = params)
    results = out[0]
    variances = out[2]
    mean_sum_squared = out[1]
    valid_time = out[3]
    preds = out[4]
    #except:
        #print("eigenvalue error occured.")

    return -1*results, mean_sum_squared, variances, valid_time, preds

def get_res_results(itr, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, traintype, \
    rktest_u_arr_train_nonoise,rktest_u_arr_test, num_tests, alpha_values, rkTime_test, split_test, system, tau, params, savepred, debug_mode):
    tic = time.perf_counter()
    print('Starting res %d' % itr)
    if debug_mode:
        reservoir = generate_res(itr, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, traintype)
    else:
        res_states_found = False
        next_itr = 0
        base_itr = np.copy(itr)
        while not res_states_found:
            try:
                reservoir = generate_res(itr, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, traintype)
                res_states_found = True
            except:
                itr = (base_itr+1)*1000+next_itr
                next_itr += 1

    toc = time.perf_counter()
    print('Res states found for itr %d, runtime: %f sec.' % (itr, toc-tic))
    #for r in reservoirs:
    #r.data_trstates = 0
    #r.states_trstates = 0
    #    get_states(r, rk, skip = 150)

    stable_frac   = np.zeros((alpha_values.size-1))
    mean_sum_squared   = np.zeros((num_tests, alpha_values.size-1))
    variances          = np.zeros((num_tests, alpha_values.size-1))
    valid_time         = np.zeros((num_tests, alpha_values.size-1))
    final_out = []
    if not isinstance(noise, np.ndarray):
        noise_array = np.array([noise])
    else:
        noise_array = noise
    for noise in noise_array:
        noise_tic = time.perf_counter()
        min_optim_func = lambda alpha: optim_func(reservoir, noise, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha, rkTime_test, split_test, traintype, system, params)
        func_vals = np.zeros(alpha_values.size)
        for j in range(alpha_values.size):
            print('Regularization: ', alpha_values[j])
            if debug_mode:
                out = min_optim_func(alpha_values[j])
                if j==0:
                    stable_frac_0 = out[0]
                    variances_0   = out[2]
                    mean_sum_squared_0 = out[1]
                    valid_time_0 = out[3]
                else:
                    stable_frac[j-1] = out[0]
                    variances[:,j-1] = out[2]
                    mean_sum_squared[:,j-1] = out[1]
                    valid_time[:,j-1] = out[3]
                if savepred:
                    if j == 1:
                        preds = out[4]
                    elif j == 2:
                        preds = np.stack((preds, out[4]), axis = 3)
                    elif j > 2:
                        preds = np.concatenate((preds, out[4].reshape(out[4].shape[0], out[4].shape[1], out[4].shape[2] ,1)), axis = 3)
            else:
                try:
                    out = min_optim_func(alpha_values[j])
                    if j==0:
                        stable_frac_0 = out[0]
                        variances_0   = out[2]
                        mean_sum_squared_0 = out[1]
                        valid_time_0 = out[3]
                    else:
                        stable_frac[j-1] = out[0]
                        variances[:,j-1] = out[2]
                        mean_sum_squared[:,j-1] = out[1]
                        valid_time[:,j-1] = out[3]
                    if savepred:
                        if j == 1:
                            preds = out[4]
                        elif j == 2:
                            preds = np.stack((preds, out[4]), axis = 3)
                        elif j > 2:
                            preds = np.concatenate((preds, out[4].reshape(out[4].shape[0], out[4].shape[1], out[4].shape[2] ,1)), axis = 3)
                except:
                    print('Training unsucessful for alpha ' + alpha_values[j])
                    if j==0:
                        stable_frac_0 = 0.0
                        variances_0   = np.zeros(num_tests)
                        mean_sum_squared_0 = np.zeros(num_tests)
                        valid_time_0 = np.zeros(num_tests)
                    else:
                        stable_frac[j-1] = 0.0
                        variances[:,j-1] = np.zeros(num_tests)
                        mean_sum_squared[:,j-1] = np.zeros(num_tests)
                        valid_time[:,j-1] = np.zeros(num_tests)
                    if savepred:
                        pred_out = np.zeros((num_tests, rktest_u_arr_test.shape[0], (rkTime_test-split_test)+1))
                        if j == 1:
                            preds = preds_out
                        elif j == 2:
                            preds = np.stack((preds, preds_out), axis = 3)
                        elif j > 2:
                            preds = np.concatenate((preds, preds_out.reshape(preds_out.shape[0], preds_out.shape[1], preds_out.shape[2] ,1)), axis = 3)
        if not savepred:
            preds = np.empty((1,1,1,1), dtype = np.complex128)

        final_out.append((stable_frac_0, stable_frac, mean_sum_squared_0, mean_sum_squared, variances_0, variances,         valid_time_0, valid_time, preds))
        noise_toc = time.perf_counter()
        print('Noise test time: %f sec.' % (noise_toc - noise_tic))
    toc = time.perf_counter()
    runtime = toc - tic
    print('Iteration runtime: %f sec.' % runtime)

    return final_out

@ray.remote
def find_stability(noisetype, noise, traintype, train_seed, res_itr, rho, sigma, leakage, win_type, bias_type, train_time, res_size, res_per_test, noise_realizations, num_tests, alpha_values, system, tau, savepred, debug_mode):
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
    elif system == 'KS':
        rkTime_test = 18000
    split_test  = 2000
    rktest_u_arr_train_nonoise, rktest_u_arr_test, params = get_test_data(tau = tau, num_tests = num_tests, rkTime = rkTime_test, split = split_test, system = system)
    np.random.seed(train_seed)
    if system == 'lorenz':
        ic = np.random.rand(3)*2-1
        rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], tau = tau, T = train_time, ttsplit = train_time, system = system, params = params)
    elif system == 'KS':
        u0 = 0.6*(np.random.rand(64)*2-1)
        rk = RungeKutta(0,0,0, tau = tau, T = train_time, ttsplit = train_time, u0 = u0, system = system, params = params)

    out = get_res_results(res_itr, rk, res_size, rho, sigma, leakage, win_type, bias_type, noisetype, noise, noise_realizations, traintype, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha_values, rkTime_test, split_test, system, tau, params, savepred, debug_mode)
    #toc_global = time.perf_counter()
    #print('Total Runtime: %s sec.' % (toc_global - tic_global))


    return out

def get_stability_output(out_full, noise_indices, train_indices, res_per_test, num_tests, alpha_values, savepred):
    noise_vals = np.unique(noise_indices)
    train_vals = np.unique(train_indices)
    tn, nt = np.meshgrid(train_vals, noise_vals)
    tn = tn.flatten()
    nt = nt.flatten()
    results = []
    for k in range(tn.size):
        out = out_full[(tn[k] == train_indices) & (nt[k] == noise_indices)]
        stable_frac_0 = np.zeros(res_per_test)
        stable_frac   = np.zeros((res_per_test, alpha_values.size-1))
        mean_sum_squared_0 = np.zeros((res_per_test, num_tests))
        mean_sum_squared   = np.zeros((res_per_test, num_tests, alpha_values.size-1))
        variances_0        = np.zeros((res_per_test, num_tests))
        variances          = np.zeros((res_per_test, num_tests, alpha_values.size-1))
        valid_time_0       = np.zeros((res_per_test, num_tests))
        valid_time         = np.zeros((res_per_test, num_tests, alpha_values.size-1))
        for i in range(res_per_test):
            stable_frac_0[i] = out[i][0]
            stable_frac[i,:]   = out[i][1]
            mean_sum_squared_0[i] = out[i][2]
            mean_sum_squared[i,:,:] = out[i][3]
            variances_0[i]   = out[i][4]
            variances[i,:,:] = out[i][5]
            valid_time_0[i]  = out[i][6]
            valid_time[i,:,:]= out[i][7]

        best_alpha_val = 0
        for j in range(1,alpha_values.size):
            if np.mean(stable_frac[:,j-1]) <= best_alpha_val:

                best_alpha_val = np.mean(stable_frac[:,j-1])
                best_stable_frac = -stable_frac[:,j-1]
                best_variances   = variances[:,:,j-1]
                best_mean_sum_squared = mean_sum_squared[:,:,j-1]
                best_valid_time  = valid_time[:,:,j-1]
                stable_frac_alpha = alpha_values[j]
                best_j = j
        if savepred:
            for i in range(res_per_test):
                if i == 0:
                    preds = out[i][8]
                elif i == 1:
                    preds = np.stack((preds, out[i][8]), axis = 0)
                elif i > 1:
                    preds = np.concatenate((preds, out[i][8].reshape(1,out[i][8].shape[0], out[i][8].shape[1], out[i][8].shape[2], out[i][8].shape[3])), axis = 0)
            if res_per_test == 1:
                preds = preds.reshape(1,preds.shape[0], preds.shape[1], preds.shape[2], preds.shape[3])

            best_preds = preds[:,:,:,:,best_j-1]
        else:
            best_preds = np.empty((1,1,1,1), dtype = np.complex128)



        result = (stable_frac_0, best_stable_frac, stable_frac_alpha, mean_sum_squared_0, best_mean_sum_squared, variances_0, best_variances, valid_time_0, best_valid_time, best_preds)

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
    system    = 'KS'
    savepred  = False
    rho       = 0.5
    sigma     = 1.0
    leakage   = 1.0
    bias_type  = 'old'
    win_type  = 'full'
    debug_mode= False
    ifray = True
    ray.init(address=os.environ["ip_head"])
    tau_flag = True

    try:
        opts, args = getopt.getopt(argv, "T:N:r:", \
                ['noisetype=','traintype=', 'system=', 'res=',\
                'tests=','trains=','savepred=', 'tau=', 'rho=',\
                'sigma=','leakage=','bias_type=', 'debug=', 'win_type='])
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
    if tau_flag:
        if system == 'lorenz':
            tau = 0.1
        elif system == 'KS':
            tau = 0.25
    if savepred:
        predflag = 'wpred_'
    else:
        predflag = ''
    ########################################
    # train_time = 500
    # res_size = 100
    # res_per_test = 100
    # noise_realizations = 1

    noise_values_array = np.logspace(-3.666666666666, 0, num = 12, base = 10)[4:8]
    #noise_values_array = np.array([np.logspace(-3.666666666666, 0, num = 12, base = 10)[5]])
    #noise_values_array = np.array([0,1e-3,1e-2])
    alpha_values = np.append(0., np.logspace(-8, -1, 15)*noise_realizations)
    #alpha_values = np.array([0,1e-6,1e-4])
    if traintype in ['gradient1','gradient2','gradient12'] or 'gradientk' in traintype:
        tr, rt = np.meshgrid(np.arange(num_trains), np.arange(res_per_test))
        tr     = tr.flatten()
        rt     = rt.flatten()
        print('Starting Ray Computation')
        tic = time.perf_counter()
        out_base  = ray.get([find_stability.remote(noisetype, noise_values_array, traintype, \
            tr[i], rt[i], rho, sigma, leakage, win_type, bias_type, train_time, \
            res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
            system, tau, savepred, debug_mode) for i in range(tr.size)])
        tnr, ntr, rtn = np.meshgrid(np.arange(num_trains), noise_values_array, np.arange(res_per_test))
        tnr = tnr.flatten()
        ntr = ntr.flatten()
        rtn = rtn.flatten()
    else:
        tnr, ntr, rtn = np.meshgrid(np.arange(num_trains), noise_values_array, np.arange(res_per_test))
        tnr = tnr.flatten()
        ntr = ntr.flatten()
        rtn = rtn.flatten()
        print('Starting Ray Computation')
        tic = time.perf_counter()
        out_base  = ray.get([find_stability.remote(noisetype, ntr[i], traintype,\
            tnr[i], rtn[i], rho, sigma, leakage, win_type, bias_type, train_time, \
            res_size, res_per_test, noise_realizations, num_tests, alpha_values,\
            system, tau, savepred, debug_mode) for i in range(tnr.size)])
    out = []
    for i in range(len(out_base)):
        for j in range(len(out_base[i])):
            out.append(out_base[i][j])

        #out = [find_stability(noisetype, ntr[i], traintype, tnr[i], rtn[i], rho, sigma, leakage, win_type, bias_type, train_time, \
        #         res_size, res_per_test, noise_realizations, num_tests, alpha_values, system, tau, savepred, debug_mode) for i in range(tnr.size)]
    results = get_stability_output(np.array(out, dtype = object), ntr, tnr, res_per_test, num_tests, alpha_values, savepred)

    tn, nt= np.meshgrid(np.arange(num_trains), noise_values_array)
    tn = tn.flatten()
    nt = nt.flatten()

    ray.shutdown()
    toc = time.perf_counter()
    runtime = toc - tic
    print('Runtime over all cores: %f sec.' %(runtime))

    stable_frac        = []
    stable_frac_0      = []
    stable_frac_alpha  = []
    mean_sum_squared   = []
    mean_sum_squared_0 = []
    variances          = []
    variances_0        = []
    valid_time_0       = []
    valid_time         = []
    mean_valid_time    = []
    for i in range(tn.size):
        stable_frac_0.append(results[i][0])
        stable_frac.append(results[i][1])
        stable_frac_alpha.append(results[i][2])
        mean_sum_squared_0.append(results[i][3])
        mean_sum_squared.append(results[i][4])
        variances_0.append(results[i][5])
        variances.append(results[i][6])
        valid_time_0.append(results[i][7])
        valid_time.append(results[i][8])
        mean_valid_time.append(np.mean(valid_time[-1]))
    if savepred:
        preds = []
        for i in range(tn.size):
            preds.append(results[i][9])
    stable_frac = np.array(stable_frac).reshape(noise_values_array.size,-1)
    stable_frac_0 = np.array(stable_frac_0).reshape(noise_values_array.size,-1)
    stable_frac_alpha = np.array(stable_frac_alpha).reshape(noise_values_array.size,-1)
    mean_valid_time = np.array(mean_valid_time).reshape(noise_values_array.size, -1)
    foldername = '/lustre/awikner1/res-noise-stabilization/'
    top_folder = '%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)
    folder = '%s_more_noisetest_%srho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, predflag, rho, sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, noisetype, traintype)
    if not os.path.isdir(os.path.join(foldername, top_folder)):
        os.mkdir(os.path.join(foldername, top_folder))
    if not os.path.isdir(os.path.join(os.path.join(foldername, top_folder), folder)):
        os.mkdir(os.path.join(os.path.join(foldername, top_folder), folder))

    # foldername = ''
    print('Ray finished')
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
        np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
            %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances_0[i], delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
            %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances[i], delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
            %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), valid_time_0[i], delimiter = ',')
        np.savetxt(foldername+top_folder+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
            %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), valid_time[i], delimiter = ',')
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
if __name__ == "__main__":
    main(sys.argv[1:])

