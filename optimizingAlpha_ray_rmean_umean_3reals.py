#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 8:00:00
#Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=60
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
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from numba import jit
import time

import pkg_resources, os
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
isray = [('ray==' in elem) for elem in installed_packages_list]
if (True in isray):
    print('Ray installed')
else:
    os.system('pip install -r -U ray')

import ray
from lorenzrungekutta_numba import rungekutta


class Reservoir:
    def __init__(self, rk, rsvr_size = 300, spectral_radius = 0.6, input_weight = 1, res_seed = 1):
        self.rsvr_size = rsvr_size

        #get spectral radius < 1
        #gets row density = 0.03333
        np.random.seed(res_seed)
        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 10/rsvr_size:
                    unnormalized_W[i][j] = 0

        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False, maxiter = 10**5)

        # self.W = sparse.csr_matrix(spectral_radius/np.abs(max_eig)*unnormalized_W)
        self.W   = spectral_radius/np.abs(max_eig)*unnormalized_W

        const_conn = int(rsvr_size*0.15)
        Win = np.zeros((rsvr_size, 4))
        Win[:const_conn, 0] = (np.random.rand(Win[:const_conn, 0].size)*2 - 1)*input_weight
        Win[const_conn: const_conn + int((rsvr_size-const_conn)/3), 1] = (np.random.rand(Win[const_conn: const_conn + int((rsvr_size-const_conn)/3), 1].size)*2 - 1)*input_weight
        Win[const_conn + int((rsvr_size-const_conn)/3):const_conn + 2*int((rsvr_size-const_conn)/3), 2] = (np.random.rand(Win[const_conn + int((rsvr_size-const_conn)/3):const_conn + 2*int((rsvr_size-const_conn)/3), 2].size)*2 - 1)*input_weight
        Win[const_conn + 2*int((rsvr_size-const_conn)/3):, 3] = (np.random.rand(Win[const_conn + 2*int((rsvr_size-const_conn)/3):, 3].size)*2 - 1)*input_weight


        """
        Win = np.zeros((rsvr_size, 3))
        q   = int(rsvr_size//3)
        for i in range(3):
            Win[q*i:q*(i+1),i] = input_weight*(np.random.rand(q)*2-1)

        # self.Win = sparse.csr_matrix(Win)
        """
        self.Win = Win
        self.X = (np.random.rand(rsvr_size, rk.train_length+2)*2 - 1)
        self.Wout = np.array([])

        self.data_trstates = np.zeros((3,self.X.shape[0]+4), dtype = np.float64)
        self.states_trstates = np.zeros((self.X.shape[0]+4,self.X.shape[0]+4), dtype = np.float64)

class RungeKutta:
    def __init__(self, x0 = 2,y0 = 2,z0 = 23, h = 0.01, T = 300, ttsplit = 5000, noise_scaling = 0, noise_seed = 10):
        u_arr = np.ascontiguousarray(rungekutta(x0,y0,z0,h,T)[:, ::10])
        self.train_length = ttsplit
        self.noise_scaling = noise_scaling

        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919

        self.u_arr_train = u_arr[:, :ttsplit+1]
        #size 5001

        #noisy training array
        #switch to gaussian
        np.random.seed(noise_seed)
        noise = np.random.randn(self.u_arr_train[:,0].size, self.u_arr_train[0,:].size)*noise_scaling
        self.u_arr_train_noise = self.u_arr_train + noise

        #plt.plot(self.u_arr_train_noise[0, :500])

        #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]
        #size 1001

@jit(nopython = True, fastmath = True)
def RungeKuttawrapped(x0 = 2,y0 = 2,z0 = 23, h = 0.01, T = 300, ttsplit = 5000, noise_scaling = 0, noise_seed = 10):
    u_arr = np.ascontiguousarray(rungekutta(x0,y0,z0,h,T)[:, ::10])
    # self.train_length = ttsplit
    # self.noise_scaling = noise_scaling

    u_arr[0] = (u_arr[0] - 0)/7.929788629895004
    u_arr[1] = (u_arr[1] - 0)/8.9932616136662
    u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919

    u_arr_train = u_arr[:, :ttsplit+1]
    #size 5001

    #noisy training array
    #switch to gaussian
    np.random.seed(noise_seed)
    noise = np.random.randn(u_arr_train[:,0].size, u_arr_train[0,:].size)*noise_scaling
    u_arr_train_noise = u_arr_train + noise

    #plt.plot(self.u_arr_train_noise[0, :500])

    #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
    u_arr_test = u_arr[:, ttsplit:]
    return u_arr_train, u_arr_train_noise, u_arr_test, ttsplit, noise_scaling

def getX(res, rk,x0 = 1,y0 = 1,z0 = 1, noise = False):
    if noise:
        u_training = rk.u_arr_train_noise
    else:
        u_training = rk.u_arr_train
    res.X = getXwrapped(np.ascontiguousarray(u_training), res.X, res.Win, res.W)

    return res.X
#takes a reservoir object res along with initial conditions

@jit(nopython = True, fastmath = True)
def getXwrapped(u_training, res_X, Win, W):

    #loops through every timestep
    for i in range(0, u_training[0].size):
        # u = np.append(1, u_training[:,i]).reshape(4,1)
        # u = u_training[:,i].reshape(3,1)
        # x = res_X[:,i].reshape(res.rsvr_size,1)
        res_X[:,i+1] = np.tanh(Win @ np.append(1., u_training[:,i])+W @ res_X[:,i])

    return res_X

def get_states(res, rk, skip = 150, noise_realizations = 1):
    res.data_trstates, res.states_trstates = get_states_wrapped(\
        np.ascontiguousarray(rk.u_arr_train), np.ascontiguousarray(rk.u_arr_train_noise), res.X, res.Win, res.W, \
        rk.noise_scaling, skip, noise_realizations)

@jit(nopython = True, fastmath = True)
def get_states_wrapped(u_arr_train, u_arr_train_noise, res_X, Win, W, noise_scaling, skip, noise_realizations):
    if noise_realizations == 1:
        Y_train = u_arr_train[:, skip+1:]
        X = getXwrapped(u_arr_train_noise, res_X, Win, W)[:, skip+1:(res_X[0].size - 1)]
        X_train = np.concatenate((np.ones((1, u_arr_train[0].size-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis = 0)
        data_trstates = Y_train @ X_train.T
        states_trstates = X_train @ X_train.T
    else:
        data_trstates = np.zeros((3,res_X.shape[0]+4), dtype = np.float64)
        states_trstates = np.zeros((res_X.shape[0]+4,res_X.shape[0]+4), dtype = np.float64)
        for i in range(noise_realizations):
            np.random.seed(i+9)
            noise = noise_scaling*np.random.randn(u_arr_train.shape[0], u_arr_train.shape[1])
            u_arr_train_noise = u_arr_train + noise

            Y_train = u_arr_train[:, skip+1:]

            X = getXwrapped(u_arr_train_noise, res_X, Win, W)[:, skip+1:(res_X[0].size - 1)]
            X_train = np.concatenate((np.ones((1, u_arr_train[0].size-(skip+1))), X, u_arr_train_noise[:, skip:-1]), axis = 0)
            if i == 0:
                X_train_mean = np.zeros(X_train.shape)
            X_train_mean += X_train/noise_realizations

        data_trstates += Y_train @ X_train_mean.T
        states_trstates += X_train_mean @ X_train_mean.T
    return data_trstates, states_trstates

def trainRRM(res, rk, skip = 150, alpha = 10**-4, noise_realizations = 1):
    print("Training... ")

    alph = alpha
    #rrm = Ridge(alpha = alph, solver = 'cholesky')
    if noise_realizations == 1:
        Y_train = rk.u_arr_train_noise[:, skip+1:]

        X = getX(res, rk, noise = True)[:, skip+1:(res.X[0].size - 1)]
        X_train = np.concatenate((np.ones((1, rk.u_arr_train[0].size-(skip+1))), X, rk.u_arr_train_noise[:, skip:-1]), axis = 0)

        data_trstates = np.matmul(Y_train, np.transpose(X_train))
        states_trstates = np.matmul(X_train,np.transpose(X_train))
        #X_train = np.copy(X)
    else:
        data_trstates = 0
        states_trstates = 0
        for i in range(noise_realizations):
            np.random.seed(i+9)
            noise = rk.noise_scaling*np.random.randn(rk.u_arr_train.shape)
            rk.u_arr_train_noise = rk.u_arr_train + noise

            Y_train = rk.u_arr_train_noise[:, skip+1:]

            X = getX(res, rk, noise = True)[:, skip+1:(res.X[0].size - 1)]
            X_train = np.concatenate((np.ones((1, rk.u_arr_train[0].size-(skip+1))), X, rk.u_arr_train_noise[:, skip:-1]), axis = 0)

            data_trstates += np.matmul(Y_train, np.transpose(X_train))
            states_trstates += np.matmul(X_train,np.transpose(X_train))

    idenmat = np.identity(res.rsvr_size+4)*alph
    res.Wout = np.transpose(solve(np.transpose(states_trstates + idenmat),np.transpose(data_trstates)))

    #optimization function needs perc. of stable res.
    #scipy.optimize.minimize
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    #function which takes log of alpha, give perc. of stable reservoirs

    #split up train function to find matrices first, second one which computed Wout
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fminbound.html
    #allows to set range of alpha

    print("Training complete ")
    #Y_train = Y_train.transpose()
    #X_train = X.transpose()

    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    #rrm.fit(X_train,Y_train)
    #res.Wout = rrm.coef_
    return res.Wout

def repeatTraining(res, T = 300, ttsplit = int(300/0.1), repeat_times = 10, skip = 150, noise_scaling = 0.1):
    ic = np.random.rand(3)*2-1
    rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = T, ttsplit = ttsplit, noise_scaling = noise_scaling)

    print("Training... ")

    alph = 10**-6
    #rrm = Ridge(alpha = alph, solver = 'cholesky')

    #train on 10 small training sets with different noise - minimize error over all
    #save the state of the reservoir for noisy datasets
    #also try - train on signal^2 or other function (get more info than just 3 vars) - no noise

    Y_train = rk.u_arr_train[:, skip+1:]
    oneTime = rk.u_arr_train[:, skip+1:]

    X = getX(res, rk, noise = True)[:, skip+1:-1]

    Y_inputs = rk.u_arr_train_noise[:, skip:(rk.u_arr_train_noise[0].size - 1)]
    for i in range(repeat_times-1):
        Y_train = np.concatenate((Y_train, oneTime), axis = 1)
        noise = np.random.randn(rk.u_arr_train[:,0].size, rk.u_arr_train[0,:].size)*noise_scaling
        rk.u_arr_train_noise = rk.u_arr_train + noise
        X = np.concatenate((X, getX(res, rk, noise = True)[:, skip+1:-1]), axis = 1)
        Y_inputs = np.concatenate((Y_inputs, rk.u_arr_train_noise[:, skip:(rk.u_arr_train_noise[0].size - 1)]), axis = 1)

    X_train = np.concatenate((np.ones((1, repeat_times*(rk.u_arr_train[0].size-(skip+1)))), X, Y_inputs), axis = 0)
    #X_train = np.copy(X)

    idenmat = np.identity(res.rsvr_size+4)*alph
    data_trstates = np.matmul(Y_train, np.transpose(X_train))
    states_trstates = np.matmul(X_train,np.transpose(X_train))
    res.Wout = np.transpose(solve(np.transpose(states_trstates + idenmat),np.transpose(data_trstates)))

    print("Training complete ")
    #Y_train = Y_train.transpose()
    #X_train = X.transpose()

    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    #rrm.fit(X_train,Y_train)
    #res.Wout = rrm.coef_
    return res.Wout

def repeatTrainingAvg(res, T = 100, ttsplit = 400, repeat_times = 10, noise_scaling = 0.01):
    rk = RungeKutta(T = T,ttsplit = ttsplit)
    Wout_final = np.zeros((3,res.rsvr_size+4))

    for i in range(repeat_times):
        noise = np.random.randn(rk.u_arr_train[:,0].size, rk.u_arr_train[0,:].size)*noise_scaling
        rk.u_arr_train_noise = rk.u_arr_train + noise
        Wout_final = np.add(Wout_final, trainRRM(res, rk, skip = 100))

    res.Wout = Wout_final/repeat_times

#CONCATENATE ALL THE DATA BEFORE RUNNING REGRESSION

def predict(res, x0 = 0, y0 = 0, z0 = 0, steps = 1000):
    Y = predictwrapped(res.X, res.Win, res.W, res.Wout, x0, y0, z0, steps)
    return Y

@jit(nopython = True, fastmath = True)
def predictwrapped(res_X, Win, W, Wout, x0, y0, z0, steps):
    Y = np.empty((3, steps + 1))
    X = np.empty((res_X.shape[0], steps + 1))

    Y[:,0] = np.array([x0,y0,z0])
    X[:,0] = res_X[:,-2]


    for i in range(0, steps):
        # y_in = Y[:,i].reshape(3,1)
        # x_prev = X[:,i].reshape(res.rsvr_size,1)
        X[:,i+1] = np.tanh(Win @ np.append(1., Y[:,i]) + W @ X[:,i])
        #X = np.concatenate((X, x_current), axis = 1)
        Y[:,i+1] = Wout @ np.concatenate((np.array([1.]), X[:,i+1], Y[:,i]))
        #y_out = np.matmul(res.Wout, x_current)
        #Y[:,i+1] = y_out


    return Y

@jit(nopython = True, fastmath = True)
def get_test_data(num_tests, rkTime, split):
    np.random.seed(0)
    ic = np.random.rand(3)*2-1
    p, u_arr_train_nonoise, u_arr_test, p, p = RungeKuttawrapped(x0 = ic[0], \
         y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)
    rktest_u_arr_train_nonoise = np.zeros((u_arr_train_nonoise.shape[0], u_arr_train_nonoise.shape[1], num_tests))
    rktest_u_arr_test = np.zeros((u_arr_test.shape[0], u_arr_test.shape[1], num_tests))
    rktest_u_arr_train_nonoise[:,:,0] = u_arr_train_nonoise
    rktest_u_arr_test[:,:,0] = u_arr_test
    for i in range(1,num_tests):
        np.random.seed(i)
        ic = np.random.rand(3)*2-1
        p, rktest_u_arr_train_nonoise[:,:,i], rktest_u_arr_test[:,:,i], p, p = RungeKuttawrapped(x0 = ic[0], \
             y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)

    return rktest_u_arr_train_nonoise, rktest_u_arr_test

def test(res, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests = 100, rkTime = 1000, split = 3000, showMapError = False, showTrajectories = False, showHist = False):
    # tic = time.perf_counter()
    stable_count, mean_sum_squared, variances = testwrapped(res.X, res.Win, res.W, res.Wout, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, rkTime, split, showMapError, showTrajectories, showHist)
    # toc = time.perf_counter()
    # runtime = toc - tic
    # print('Test time: %f sec.' % runtime)
    return stable_count/num_tests, mean_sum_squared, variances

@jit(nopython = True, fastmath = True)
def testwrapped(res_X, Win, W, Wout, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, rkTime, split, showMapError = True, showTrajectories = True, showHist = True):

    stable_count = 0
    valid_time = np.zeros(num_tests)
    max_sum_square = np.zeros(num_tests)
    mean_sum_square = np.zeros(num_tests)
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)

    for i in range(num_tests):

        vtchange = 0

        """
        np.random.seed(i)
        ic = np.random.rand(3)*2-1
        u_arr_train, u_arr_train_nonoise, u_arr_test, train_length, noise_scaling = \
            RungeKuttawrapped(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)
        """
        res_X = (np.zeros((res_X.shape[0], split+2))*2 - 1)

        #sets res.X
        res_X = getXwrapped(np.ascontiguousarray(rktest_u_arr_train_nonoise[:,:,i]), res_X, Win, W)

        pred = predictwrapped(res_X, Win, W, Wout, x0 = rktest_u_arr_test[0,0,i], y0 = rktest_u_arr_test[1,0,i], z0 = rktest_u_arr_test[2,0,i], steps = (int(rkTime/0.1)-split))
        lorenz_map_x = np.zeros(pred[0].size)
        lorenz_map_x[0] = pred[0][0]
        x2y2z2 = np.zeros(pred.shape[1])

        check_vt = True
        for j in range(0, pred[0].size):
            if (j > 0):
                # vtchange = vtchange + (u_arr_test[0, j] - u_arr_test[0, j-1])**2 + (u_arr_test[1, j] - u_arr_test[1, j-1])**2 + (u_arr_test[2, j] - u_arr_test[2, j-1])**2
                vtchange += np.sum((rktest_u_arr_test[:,j,i]-rktest_u_arr_test[:,j-1,i])**2)

                rkmap_u_arr_train = RungeKuttawrapped(pred[0][j-1]*7.929788629895004, pred[1][j-1]*8.9932616136662, pred[2][j-1]*8.575917849311919+23.596294463016896, h=0.01, T=0.1)[0]
                lorenz_map_x[j] = rkmap_u_arr_train[0][1]

                #EXAMINE!!!
                # x2error = (pred[0][j]-rkmap_u_arr_train[0][1])**2
                # y2error = (pred[1][j]-rkmap_u_arr_train[1][1])**2
                # z2error = (pred[2][j]-rkmap_u_arr_train[2][1])**2

                # x2y2z2 = np.append(x2y2z2, (x2error+y2error+z2error))
                x2y2z2[j] = np.sum((pred[:,j]-rkmap_u_arr_train[:,1])**2)

            if (np.abs(pred[0, j] - rktest_u_arr_test[0, j,i]) > 1.5) and check_vt:
                valid_time[i] = j

                # print("Test " + str(i) + " valid time: " + str(j))
                check_vt = False

        x2y2z2 = x2y2z2/1.45
        #print(vtchange/(pred[0].size-1))
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
            #plt.plot(lorenz_map_x, label = "Map Trajectory", color = "green")
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

        means[i] = np.mean(pred[0])
        variances[i] = np.var(pred[0])

        if np.mean(x2y2z2) < 0.01 and 0.98 < np.var(pred[0]) and np.var(pred[0]) < 1.01:
            stable_count += 1
            # print("stable")
            # print()
        # else:
            # print("unstable")
            # print()


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


    return stable_count, mean_sum_square, variances

def generate_res(num_res, rk, res_size, noise_realizations = 1):

    reservoirs = []
    itr = 0

    while len(reservoirs) < num_res:
        try:
            itr += 1
            reservoirs.append(Reservoir(rk, rsvr_size = res_size, spectral_radius = 0.5, input_weight = 1.0, res_seed = itr))
            get_states(reservoirs[-1], rk, skip = 150, noise_realizations = noise_realizations)
        except:
            print("eigenvalue error occured.")
    return reservoirs

def optim_func(reservoirs, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha, rkTime = 400, split = 2000):

    results = np.zeros(len(reservoirs))
    variances = np.zeros((len(reservoirs),num_tests))
    mean_sum_squared = np.zeros((len(reservoirs),num_tests))

    for i, res in enumerate(reservoirs):
        #try:
        idenmat = np.identity(res.rsvr_size+4)*alpha
        res.Wout = np.transpose(solve(np.transpose(res.states_trstates + idenmat),np.transpose(res.data_trstates)))
        out =  test(res, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests = num_tests, rkTime = rkTime, split =split,  \
             showMapError = True, showTrajectories = True, showHist = True)
        results[i] = out[0]
        variances[i] = out[2]
        mean_sum_squared[i] = out[1]
        #except:
            #print("eigenvalue error occured.")

    return -1*np.mean(results), mean_sum_squared, variances

def trainAndTest(alph):

    results = np.array([])
    num_res = 10

    for i in range(num_res):
        try:
            ic = np.random.rand(3)*2-1
            rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = 500, ttsplit = int(500/0.1), noise_scaling = 0.01)
            res = Reservoir(rk, rsvr_size = 100, spectral_radius = 0.5, input_weight = 1.0)

            trainRRM(res, rk, skip = 150, alpha = alph)

            results = np.append(results, test(res, 1, rkTime = 400, split = 2000, showMapError = False, showTrajectories = False, showHist = False))
        except:
            print("eigenvalue error occured.")
    return -1*np.mean(results)

@ray.remote
def find_stability(noise, train_seed, train_time, res_size, res_per_test, noise_realizations, num_tests, alpha_values):
    rkTime_test = 400
    split_test  = 2000
    rktest_u_arr_train_nonoise, rktest_u_arr_test = get_test_data(num_tests = num_tests, rkTime = rkTime_test, split = split_test)

    np.random.rand(train_seed)
    ic = np.random.rand(3)*2-1
    rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = train_time, ttsplit = int(train_time/0.1), noise_scaling = noise)

    reservoirs = generate_res(res_per_test, rk, res_size = res_size, noise_realizations = noise_realizations)
        #for r in reservoirs:
            #r.data_trstates = 0
            #r.states_trstates = 0
        #    get_states(r, rk, skip = 150)

    min_optim_func = lambda alpha: optim_func(reservoirs, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha, rkTime_test, split_test)
    func_vals = np.zeros(alpha_values.size)
    for j in range(alpha_values.size):
        out = min_optim_func(alpha_values[j])
        func_vals[j] = out[0]
        if j==0:
            variances_0 = out[2]
            mean_sum_squared_0 = out[1]
        elif j == 1:
            variances = out[2]
            mean_sum_squared = out[1]
        elif func_vals[j] < np.min(func_vals[1:j]):
            variances = out[2]
            mean_sum_squared = out[1]
    stable_frac_0 = -func_vals[0]
    result_fun = np.min(func_vals[1:])
    result_x   = alpha_values[np.argmin(func_vals[1:])]
    stable_frac = -result_fun
    stable_frac_alpha = result_x

    return stable_frac_0, stable_frac, stable_frac_alpha, mean_sum_squared_0, mean_sum_squared, variances_0, variances


def main(argv):
    train_time = 50
    res_size = 100
    res_per_test = 50
    noise_realizations = 3
    num_tests = 50
    num_trains = 20
    num_procs = 60
    ifray = True
    ray.init(num_cpus = num_procs)

    try:
        opts, args = getopt.getopt(argv, "T:N:r:", [])
    except getopt.GetoptError:
        print('Error: Some options not recognized')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-T':
            train_time = int(arg)
            print(train_time)
        elif opt == '-N':
            res_size = int(arg)
            print(res_size)
        elif opt == '-r':
            noise_realizations = int(arg)
            print(noise_realizations)
    ########################################
    # train_time = 500
    # res_size = 100
    # res_per_test = 100
    # noise_realizations = 1

    noise_values_array = np.logspace(-3.666666666666, 0, num = 12, base = 10)
    alpha_values = np.append(0., np.logspace(-8, -2, 13))
    tn, nt = np.meshgrid(np.arange(num_trains), noise_values_array)
    tn = tn.flatten()
    nt = nt.flatten()
    #alpha_values = np.logspace(-8,-2,2)
    print('Starting Ray Computation')
    tic = time.perf_counter()
    results = [find_stability.remote(nt[i], tn[i], train_time, \
            res_size, res_per_test, noise_realizations, num_tests, alpha_values) for i in range(tn.size)]
    results = ray.get(results)
    toc = time.perf_counter()
    runtime = toc - tic
    print('Runtime with %d cores: %f sec.' %(num_procs, runtime))

    stable_frac = []
    stable_frac_0 = []
    stable_frac_alpha = []
    mean_sum_squared = []
    mean_sum_squared_0 = []
    variances = []
    variances_0 = []
    for i in range(tn.size):
        stable_frac_0.append(results[i][0])
        stable_frac.append(results[i][1])
        stable_frac_alpha.append(results[i][2])
        mean_sum_squared_0.append(results[i][3])
        mean_sum_squared.append(results[i][4])
        variances_0.append(results[i][5])
        variances.append(results[i][6])
    stable_frac = np.array(stable_frac).reshape(noise_values_array.size,-1)
    stable_frac_0 = np.array(stable_frac_0).reshape(noise_values_array.size,-1)
    stable_frac_alpha = np.array(stable_frac_alpha).reshape(noise_values_array.size,-1)
    foldername = '/lustre/awikner1/res-noise-stabilization/'
    folder = 'noisetest_%dnodes_%dtrain_%dreals_rmean/' % (res_size, train_time, noise_realizations)
    if not os.path.isdir(os.path.join(foldername, folder)):
        os.mkdir(os.path.join(foldername, folder))

    # foldername = ''
    """
    noise_values = np.logspace(-3.666666666666, 0, num = 12, base = 10)
    alpha_values = np.logspace(-8, -2, 13)
    stable_frac  = np.zeros(noise_values.size)
    stable_frac_alpha  = np.zeros(noise_values.size)

    foldername = '/lustre/awikner1/res-noise-stabilization/'
    # foldername = ''


    #rk = RungeKutta(x0 = 1, y0 = 1, z0 = 30, T = train_time, ttsplit = int(train_time/0.1), noise_scaling = 0.01)
    #reservoirs = generate_res(res_per_test, rk, res_size = res_size)
    rkTime_test = 400
    split_test  = 2000
    rktest_u_arr_train_nonoise, rktest_u_arr_test = get_test_data(num_tests = num_tests, rkTime = rkTime_test, split = split_test)

    for i, noise in enumerate(noise_values):
        ic = np.random.rand(3)*2-1
        rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = train_time, ttsplit = int(train_time/0.1), noise_scaling = noise)

        reservoirs = generate_res(res_per_test, rk, res_size = res_size, noise_realizations = noise_realizations)
        #for r in reservoirs:
            #r.data_trstates = 0
            #r.states_trstates = 0
        #    get_states(r, rk, skip = 150)

        min_optim_func = lambda alpha: optim_func(reservoirs, rktest_u_arr_train_nonoise, rktest_u_arr_test, num_tests, alpha, rkTime_test, split_test)
        func_vals = np.zeros(alpha_values.size)
        for j in range(alpha_values.size):
            func_vals[j] = min_optim_func(alpha_values[j])
        result_fun = np.min(func_vals)
        result_x   = alpha_values[np.argmin(func_vals)]
        stable_frac[i] = -result_fun
        stable_frac_alpha[i] = result_x
    """
    print('Ray finished')
    np.savetxt(foldername+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest_noreg.csv' %(res_size, train_time, noise_realizations), stable_frac_0,delimiter = ',')
    np.savetxt(foldername+folder+'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' %(res_size, train_time, noise_realizations), stable_frac, delimiter = ',')
    np.savetxt(foldername+folder+'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' %(res_size, train_time, noise_realizations), stable_frac_alpha, delimiter = ',')
    noise_values_array = noise_values_array.flatten()
    for i in range(nt.size):
        np.savetxt(foldername+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
                 %(res_size, train_time, noise_realizations, nt[i],tn[i]+1), mean_sum_squared_0[i]  , delimiter = ',')
        np.savetxt(foldername+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                %(res_size, train_time, noise_realizations, nt[i],tn[i]+1), mean_sum_squared[i]  , delimiter = ',')
        np.savetxt(foldername+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d_noreg.csv' \
              %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances_0[i], delimiter = ',')
        np.savetxt(foldername+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
             %(res_size, train_time, noise_realizations, nt[i], tn[i]+1), variances[i], delimiter = ',')
if __name__ == "__main__":
    main(sys.argv[1:])
