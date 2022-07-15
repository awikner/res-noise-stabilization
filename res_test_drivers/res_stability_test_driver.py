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
rho = 0.6
sigma = 0.1
leakage = 1.0

discard_time = 500
traintypes   = ['normal','gradientk1','normal','gradientk1','normal','gradientk4','regzerok4']
train_time   = 20000
res_size     = 500
noise_realization = 1
noisetypes   = ['none','none','none','none','gaussian','none','none']
tau          = 0.25
win_type     = 'full_0centered'
squarenodes  = True
system       = 'KS'
bias_type    = 'new_random'
noise_values_array_all = [np.array([0.0]),\
                      np.array([10.**(-6.8)]),\
                      np.array([0.0]),\
                      np.array([10.**(-5.4)]),\
                      np.array([10.**(-7.4)]),\
                      np.array([10.**(-7.4)]),\
                      np.array([10.**(-7.4)])]
reg_values_all         = [np.array([0.0]),\
                      np.array([0.0]),\
                      np.array([10.**(-6.0)]),\
                      np.array([10.**(-8.5)]),\
                      np.array([10.**(-16)]),\
                      np.array([10.**(-16)]),\
                      np.array([10.**(-16.5)])]

test_time       = 18000
return_all      = True
savepred        = False
save_time_rms   = False
debug_mode      = False
ifray           = False
just_process    = False
import_res      = False
import_train    = False
import_test     = False
import_noise    = False
nojit           = False
res_per_test    = 20
num_trains      = 10
num_tests       = 5
cpus_per_node   = 1
metric          = 'mss_var'
max_valid_time  = 2000
prior           = 'zero'
save_eigenvals  = False
pmap            = False
reg_train_times_all = [np.array([20000]),\
                   np.array([20000]),\
                   np.array([20000]),\
                   np.array([20000]),\
                   np.array([20000]),\
                   np.array([20000]),\
                   np.array([4])]
set_numba(os.getcwd(),nojit)
for noisetype, traintype, noise_values_array, reg_values, reg_train_times in zip(noisetypes, traintypes, noise_values_array_all, reg_values_all, reg_train_times_all): 
    run_opts = RunOpts(system = system, traintype = traintype, noisetype = noisetype, noise_realizations = noise_realizations,\
            res_size = res_size, train_time = train_time, test_time = test_time, rho = rho, sigma = sigma, leakage = leakage,\
            tau = tau, win_type = win_type, bias_type = bias_type, noise_values_array = noise_values_array,\
            savepred = savepred, save_time_rms = save_time_rms, squarenodes = squarenode, debug_mode = debug_mode,\
            res_per_test = res_per_test, num_trains = num_trains, num_tests = num_tests,\
            metric = metric, pmap = pmap, return_all = return_all,\
            max_valid_time = max_valid_time, ifray = ifray,\
            import_res = import_res,import_train = import_train, import_test =import_test, import_noise = import_noise,\
            reg_values = reg_values, reg_train_times = reg_train_times_in, discard_time = discard_time,\
            prior = prior, save_eigenvals = save_eigenvals)
    if not just_process:
        start_reservoir_test(run_opts=run_opts)
        process_data(run_opts=run_opts)


