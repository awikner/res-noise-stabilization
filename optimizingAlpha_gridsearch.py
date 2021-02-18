#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:57:49 2020

@author: josephharvey
"""
from datetime import datetime
from lorenzrungekutta_numba import rungekutta
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
        
        self.W = sparse.csr_matrix(spectral_radius/np.abs(max_eig)*unnormalized_W)
        
        const_conn = int(rsvr_size*0.15)
        Win = np.zeros((rsvr_size, 4))
        Win[:const_conn, 0] = (np.random.rand(Win[:const_conn, 0].size)*2 - 1)*input_weight
        Win[const_conn: const_conn + int((rsvr_size-const_conn)/3), 1] = (np.random.rand(Win[const_conn: const_conn + int((rsvr_size-const_conn)/3), 1].size)*2 - 1)*input_weight
        Win[const_conn + int((rsvr_size-const_conn)/3):const_conn + 2*int((rsvr_size-const_conn)/3), 2] = (np.random.rand(Win[const_conn + int((rsvr_size-const_conn)/3):const_conn + 2*int((rsvr_size-const_conn)/3), 2].size)*2 - 1)*input_weight
        Win[const_conn + 2*int((rsvr_size-const_conn)/3):, 3] = (np.random.rand(Win[const_conn + 2*int((rsvr_size-const_conn)/3):, 3].size)*2 - 1)*input_weight
        
        self.Win = sparse.csr_matrix(Win)
        self.X = (np.random.rand(rsvr_size, rk.train_length+2)*2 - 1)
        self.Wout = np.array([])

        self.data_trstates = 0
        self.states_trstates = 0
        
class RungeKutta:
    def __init__(self, x0 = 2,y0 = 2,z0 = 23, h = 0.01, T = 300, ttsplit = 5000, noise_scaling = 0):
        u_arr = rungekutta(x0,y0,z0,h,T)[:, ::10] 
        self.train_length = ttsplit
        
        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        
        self.u_arr_train = u_arr[:, :ttsplit+1] 
        #size 5001
        
        #noisy training array
        #switch to gaussian 
        noise = np.random.randn(self.u_arr_train[:,0].size, self.u_arr_train[0,:].size)*noise_scaling 
        self.u_arr_train_noise = self.u_arr_train + noise
        
        #plt.plot(self.u_arr_train_noise[0, :500])
        
        #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]
        #size 1001
    
#takes a reservoir object res along with initial conditions
def getX(res, rk,x0 = 1,y0 = 1,z0 = 1, noise = False):
    
    if noise:
        u_training = rk.u_arr_train_noise
    else:
        u_training = rk.u_arr_train
    
    #loops through every timestep
    for i in range(0, u_training[0].size):
        u = np.append(1, u_training[:,i]).reshape(4,1)
        
        x = res.X[:,i].reshape(res.rsvr_size,1)
        x_update = np.tanh(np.add(res.Win.dot(u), res.W.dot(x)))
        
        res.X[:,i+1] = x_update.reshape(1,res.rsvr_size)    
    
    return res.X

def get_states(res, rk, skip = 150):
    Y_train = rk.u_arr_train_noise[:, skip+1:]
    X = getX(res, rk, noise = True)[:, skip+1:(res.X[0].size - 1)]
    X_train = np.concatenate((np.ones((1, rk.u_arr_train[0].size-(skip+1))), X, rk.u_arr_train_noise[:, skip:-1]), axis = 0)
    res.data_trstates = np.matmul(Y_train, np.transpose(X_train))
    res.states_trstates = np.matmul(X_train,np.transpose(X_train))

    return 
    
def trainRRM(res, rk, skip = 150, alpha = 10**-4):
    print("Training... ")

    alph = alpha
    #rrm = Ridge(alpha = alph, solver = 'cholesky')
    
    Y_train = rk.u_arr_train_noise[:, skip+1:]

    
    X = getX(res, rk, noise = True)[:, skip+1:(res.X[0].size - 1)]
    X_train = np.concatenate((np.ones((1, rk.u_arr_train[0].size-(skip+1))), X, rk.u_arr_train_noise[:, skip:-1]), axis = 0) 
    #X_train = np.copy(X)
    
    idenmat = np.identity(res.rsvr_size+4)*alph
    data_trstates = np.matmul(Y_train, np.transpose(X_train))
    states_trstates = np.matmul(X_train,np.transpose(X_train))
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
    Y = np.empty((3, steps + 1))
    X = np.empty((res.rsvr_size, steps + 1))
    
    Y[:,0] = np.array([x0,y0,z0]).reshape(1,3) 
    X[:,0] = res.X[:,-2]

    
    for i in range(0, steps):
        y_in = np.append(1, Y[:,i]).reshape(4,1)
        x_prev = X[:,i].reshape(res.rsvr_size,1)
        
        x_current = np.tanh(np.add(res.Win.dot(y_in), res.W.dot(x_prev)))
        X[:,i+1] = x_current.reshape(1,res.rsvr_size)
        #X = np.concatenate((X, x_current), axis = 1)
        y_out = np.matmul(res.Wout, np.concatenate((np.array([[1]]), x_current, Y[:,i].reshape(3,1)), axis = 0))
        #y_out = np.matmul(res.Wout, x_current)
        Y[:,i+1] = y_out.reshape(1, 3)
        

    return Y

def test(res, num_tests = 100, rkTime = 1000, split = 3000, showMapError = True, showTrajectories = True, showHist = True):

    stable_count = 0
    valid_time = np.array([])
    max_sum_square = np.array([])
    mean_sum_square = np.array([]) 
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    
    for i in range(num_tests):
        
        vtchange = 0
        x2y2z2 = np.array([])
    
        ic = np.random.rand(3)*2-1
        rktest = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)
        res.X = (np.zeros((res.rsvr_size, split+2))*2 - 1)
        
        #sets res.X
        getX(res, rktest)
        
        pred = predict(res, x0 = rktest.u_arr_test[0,0], y0 = rktest.u_arr_test[1,0], z0 = rktest.u_arr_test[2,0], steps = (int(rkTime/0.1)-split))
        lorenz_map_x = np.zeros(pred[0].size)
        lorenz_map_x[0] = pred[0][0]
        
        check_vt = True
        for j in range(0, pred[0].size):
            if (j > 0):
                vtchange = vtchange + (rktest.u_arr_test[0, j] - rktest.u_arr_test[0, j-1])**2 + (rktest.u_arr_test[1, j] - rktest.u_arr_test[1, j-1])**2 + (rktest.u_arr_test[2, j] - rktest.u_arr_test[2, j-1])**2
                
                rkmap = RungeKutta(pred[0][j-1]*7.929788629895004, pred[1][j-1]*8.9932616136662, pred[2][j-1]*8.575917849311919+23.596294463016896, h=0.01, T=0.1)
                lorenz_map_x[j] = rkmap.u_arr_train[0][1] 
                
                #EXAMINE!!!
                x2error = (pred[0][j]-rkmap.u_arr_train[0][1])**2
                y2error = (pred[1][j]-rkmap.u_arr_train[1][1])**2
                z2error = (pred[2][j]-rkmap.u_arr_train[2][1])**2
                
                x2y2z2 = np.append(x2y2z2, (x2error+y2error+z2error)) 
                
            if (np.abs(pred[0, j] - rktest.u_arr_test[0, j]) > 1.5) and check_vt:
                valid_time = np.append(valid_time, j)
                
                print("Test " + str(i) + " valid time: " + str(j))
                check_vt = False
        
        x2y2z2 = x2y2z2/1.45
        #print(vtchange/(pred[0].size-1)) 
        #print("Mean: " + str(np.mean(pred[0])))
        #print("Variance: " + str(np.var(pred[0])))
        
        if showHist:
            plt.figure() 
            plt.hist(pred[0], bins = 11, label = "Predictions", alpha = 0.75)
            plt.hist(rktest.u_arr_test[0], bins = 11, label = "Truth", alpha = 0.75)
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
            plt.plot(rktest.u_arr_test[0], label = "Truth") 
            plt.legend(loc="upper right") 
        
        print("Variance of lorenz data x dim: " + str(np.var(rktest.u_arr_test[0])))
        print("Variance of predictions: " + str(np.var(pred[0]))) 
        print("Max of total square error: " + str(max(x2y2z2)))
        print("Mean of total error: " + str(np.mean(x2y2z2)))
        print("Wasserstein distance: " + str(wasserstein_distance(pred[0], rktest.u_arr_test[0])))
        print()
        
        max_sum_square = np.append(max_sum_square, max(x2y2z2))
        mean_sum_square = np.append(mean_sum_square, np.mean(x2y2z2)) 
        
        means[i] = np.mean(pred[0])
        variances[i] = np.var(pred[0])
        
        if np.mean(x2y2z2) < 0.01 and 0.98 < np.var(pred[0]) and np.var(pred[0]) < 1.01:
            stable_count += 1
            print("stable")
            print()
        else:
            print("unstable")
            print() 
        
    
    
    if showMapError or showTrajectories or showHist:
        plt.show()
    
    #print("Variance of total square error: " + str(np.var(x2y2z2)))

    print("Avg. max sum square: " + str(np.mean(max_sum_square)))
    print("Avg. mean sum square: " + str(np.mean(mean_sum_square))) 
    print("Avg. of x dim: " + str(np.mean(means)))
    print("Var. of x dim: " + str(np.mean(variances)))
    print()
    
    
    return stable_count/num_tests

def generate_res(num_res, rk, res_size):

    reservoirs = []

    while len(reservoirs) < num_res:
        try:
            reservoirs.append(Reservoir(rk, rsvr_size = res_size, spectral_radius = 0.5, input_weight = 1.0))
            get_states(reservoirs[-1], rk, skip = 150) 
        except:
            print("eigenvalue error occured.")
    return reservoirs

def optim_func(reservoirs, alpha):

    results = np.array([])

    for res in reservoirs: 
        try:
            idenmat = np.identity(res.rsvr_size+4)*alpha
            res.Wout = np.transpose(solve(np.transpose(res.states_trstates + idenmat),np.transpose(res.data_trstates))) 

            results = np.append(results, test(res, 1, rkTime = 400, split = 2000, showMapError = False, showTrajectories = False, showHist = False))
        except:
            print("eigenvalue error occured.")

    return -1*np.mean(results)

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



########################################
np.random.seed(0)

train_time = 500
res_size = 100
res_per_test = 100
noise_values = np.logspace(-3.666666666666, 0, num = 12, base = 10)
alpha_values = np.logspace(-8, -2, 13)
stable_frac  = np.zeros(noise_values.size)
stable_frac_alpha  = np.zeros(noise_values.size)


#rk = RungeKutta(x0 = 1, y0 = 1, z0 = 30, T = train_time, ttsplit = int(train_time/0.1), noise_scaling = 0.01)
#reservoirs = generate_res(res_per_test, rk, res_size = res_size)

for i, noise in enumerate(noise_values):
    ic = np.random.rand(3)*2-1
    rk = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = train_time, ttsplit = int(train_time/0.1), noise_scaling = noise)
    
    reservoirs = generate_res(res_per_test, rk, res_size = res_size)
    #for r in reservoirs:
        #r.data_trstates = 0
        #r.states_trstates = 0 
    #    get_states(r, rk, skip = 150)

    min_optim_func = lambda alpha: optim_func(reservoirs, alpha)
    func_vals = np.zeros(alpha_values.size)
    for j in range(alpha_values.size):
        func_vals[j] = min_optim_func(alpha_values[j])
    result_fun = np.min(func_vals)
    result_x   = alpha_values[np.argmin(func_vals)]
    stable_frac[i] = -result_fun    
    stable_frac_alpha[i] = result_x
    f = open("noise varying data", "a")
    now = datetime.now()
    currenttime = now.strftime("%H:%M:%S")
    f.write(currenttime)
    f.write("\n")
    f.write("noise level: " + str(noise))
    f.write("\n")
    f.write("res size: " + str(res_size))
    f.write("\n")
    f.write("train time: " + str(train_time))
    f.write("\n")
    f.write("max fraction of stable reservoirs: " + str(-1*result_fun))
    f.write("\n")
    f.write("optimal alpha: " + str(result_x))
    f.write("\n")
    f.write("\n")
    f.close()

np.savetxt('stable_frac_%dnodes_%dtrainiters.csv' %(res_size, train_time), stable_frac, delimiter = ',')
np.savetxt('stable_frac_alpha_%dnodes_%dtrainiters.csv' %(res_size, train_time), stable_frac_alpha, delimiter = ',')














