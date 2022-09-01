import sys,os
from res_reg_lmnt_awikner.helpers import set_numba
from src.res_reg_lmnt_awikner.reservoir_train_test import start_reservoir_test
from src.res_reg_lmnt_awikner.process_test_data import process_data
from src.res_reg_lmnt_awikner.classes import RunOpts, ResData

import subprocess
import numpy as np
import time
from itertools import product
import re
rho = 0.6
sigma = 0.1
leakage = 1.0

discard_time = 500
traintypes   = ['normal','gradientk1','normal','gradientk1','normal','gradientk4','regzerok4']
#traintypes = ['normal','gradientk1','normal']
train_time   = 20000
res_size     = 500
noise_realizations = 1
noisetypes   = ['none','none','none','none','gaussian','none','none']
tau          = 0.25
win_type     = 'full_0centered'
squarenodes  = True
system       = 'KS'
bias_type    = 'new_random'
noise_values_array_all = [np.array([0.0]),
                      np.array([10.**(-6.8)]),
                      np.array([0.0]),
                      np.array([10.**(-5.4)]),
                      np.array([10.**(-7.4)]),
                      np.array([10.**(-7.4)]),
                      np.array([10.**(-7.4)])]
reg_values_all         = [np.array([0.0]),
                      np.array([0.0]),
                      np.array([10.**(-6.0)]),
                      np.array([10.**(-8.5)]),
                      np.array([10.**(-16)]),
                      np.array([10.**(-16)]),
                      np.array([10.**(-16.5)])]

test_time       = 16000
return_all      = True
savepred        = False
save_time_rms   = False
debug_mode      = True
ifray           = True
just_process    = False
just_display    = False
nojit           = False
#res_per_test    = 20
res_per_test    = 4
#num_trains      = 10
num_trains      = 3
num_tests       = 5
cpus_per_node   = 4
metric          = 'mss_var'
machine         = 'personal'
max_valid_time  = 2000
prior           = 'zero'
save_eigenvals  = False
pmap            = False
reg_train_times_all = [np.array([20000]),
                   np.array([20000]),
                   np.array([20000]),
                   np.array([20000]),
                   np.array([20000]),
                   np.array([20000]),
                   np.array([4])]
set_numba(os.getcwd(),nojit)
all_data = []
all_run_opts = []
for i, (noisetype, traintype, noise_values_array, reg_values, reg_train_times_in) in enumerate(zip(noisetypes, traintypes, noise_values_array_all, reg_values_all, reg_train_times_all)):
    run_opts = RunOpts(system = system, traintype = traintype, noisetype = noisetype, noise_realizations = noise_realizations,
            res_size = res_size, train_time = train_time, test_time = test_time, rho = rho, sigma = sigma, leakage = leakage,
            tau = tau, win_type = win_type, bias_type = bias_type, noise_values_array = noise_values_array,
            savepred = savepred, save_time_rms = save_time_rms, squarenodes = squarenodes, debug_mode = debug_mode,
            res_per_test = res_per_test, num_trains = num_trains, num_tests = num_tests,
            metric = metric, pmap = pmap, return_all = return_all, machine = machine,
            max_valid_time = max_valid_time, ifray = ifray,
            reg_values = reg_values, reg_train_times = reg_train_times_in, discard_time = discard_time,
            prior = prior, save_eigenvals = save_eigenvals, num_cpus = cpus_per_node)
    if not just_process:
        start_reservoir_test(run_opts=run_opts)
        time.sleep(5)
    if not just_display:
        process_data(run_opts=run_opts)
    all_data.append(ResData(run_opts))
    all_run_opts.append(run_opts)

lyapunov_time = 1./tau/0.048
gross_err_bnd = 1.0

for data, run_opts in zip(all_data, all_run_opts):
    data_slice = data.data_slice(reg = run_opts.reg_values[0], reg_train = run_opts.reg_train_times[0],
                                 noise = run_opts.noise_values_array[0], median_flag = True, metric = 'gross_frac',
                                 gross_err_bnd = gross_err_bnd, reduce_axes = ['res','train','test'])
    if run_opts.reg_values[0] == 0.:
        reg_str = '0.0'
    else:
        reg_str = '10^(%0.1f)' % np.log10(run_opts.reg_values[0])
    if run_opts.noise_values_array[0] == 0.:
        noise_str = '0.0'
    else:
        noise_str = '10^(%0.1f)' % np.log10(run_opts.noise_values_array[0])
    print('Noisetype: %s, Traintype: %s, Tikhonov: %s, Other Reg.: %s' %\
            (run_opts.noisetype, run_opts.traintype, reg_str, noise_str))
    print('Fraction Stable: %0.3f, Median VT: %0.2f, Median Mean Map error: %0.2e, Median Max Map Error: %0.2e' %\
            (1.0-data_slice['gross_frac'].iloc[0], data_slice['valid_time'].iloc[0]/lyapunov_time,
            data_slice['mean_rms'].iloc[0], data_slice['max_rms'].iloc[0]))
    print('')
