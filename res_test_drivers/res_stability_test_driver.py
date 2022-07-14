import sys,os
sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(),'helpers'))
sys.path.insert(1, os.path.join(os.getcwd(),'data_scripts'))
import subprocess
import numpy as np
import time
from itertools import product
import re
from helpers.set_numba import *
from reservoir_train_test import *
from data_scripts.process_test_data import *
rhos = np.array([0.6])
sigmas = np.array([0.1])
leakages = np.array([1.0])
rhos_mat, sigmas_mat, leakages_mat = np.meshgrid(rhos, sigmas, leakages)
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

discard_time = 500
traintypes = np.array(['normal'])
train_times = np.array([21000]*traintypes.size)
res_sizes = np.array([500]*traintypes.size)
nos = np.array([1]*traintypes.size)
noisetypes = ['gaussian']
taus       = [0.25]*len(traintypes)
win_types  = ['full_0centered']*len(taus)
squarenodes = [True]*len(taus)
system     = 'KS'

bias_type = 'new_random'
noise_values_array_all = [np.array([np.logspace(-8,-3,num=26, base = 10)[11]])]
reg_values_all = [np.logspace(-9,-8,3)]

test_time = 4000
return_all  = True
savepred   = False
save_time_rms   = False
debug_mode      = True
debug_part = True
ifray   = False
just_process = False
import_res = False
import_train = False
import_test = False
import_noise = False
nojit = False
res_per_test    = 1
num_trains = 1
num_tests  = 4
num_nodes = 1
cpus_per_node = 6
runtime    = '15:00'
metric     = 'mss_var'
account    = 'physics-hi'
max_valid_time = 2000
prior = 'zero'
save_eigenvals = False
pmap = False
reg_train_times_in = None
set_numba(os.getcwd(),nojit)
for i, (noisetype, res_size, win_type, traintype, noise_realizations, train_time, tau, squarenode) in enumerate(zip(noisetypes, res_sizes, win_types, traintypes, nos, train_times, taus, squarenodes)):

    for rho, leakage, sigma in zip(rhos, leakages, sigmas):
        
        run_opts = RunOpts(system = system, traintype = traintype, noisetype = noisetype, noise_realizations = noise_realizations,\
            res_size = res_size, train_time = train_time, test_time = test_time, rho = rho, sigma = sigma, leakage = leakage,\
            tau = tau, win_type = win_type, bias_type = bias_type, noise_values_array = noise_values_array_all[i],\
            savepred = savepred, save_time_rms = save_time_rms, squarenodes = squarenode, debug_mode = debug_mode,\
            res_per_test = res_per_test, num_trains = num_trains, num_tests = num_tests,\
            metric = metric, pmap = pmap, return_all = return_all,\
            max_valid_time = max_valid_time, ifray = ifray,\
            import_res = import_res,import_train = import_train, import_test =import_test, import_noise = import_noise,\
            reg_values = reg_values_all[i], reg_train_times = reg_train_times_in, discard_time = discard_time,\
            prior = prior, save_eigenvals = save_eigenvals)
        start_reservoir_test(run_opts=run_opts)
        process_data(run_opts=run_opts)

