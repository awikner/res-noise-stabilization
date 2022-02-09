import warnings
from itertools import product
import sys
import getopt
import os

import numpy as np


def get_run_opts(argv, runflag = True):
    #print(argv)

    if runflag:
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
        save_time_rms = False
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
                    'machine=', 'num_cpus=', 'pmap=', 'parallel=', 'metric=','returnall=',
                    'savetime='])
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
            elif opt == '--savetime':
                if str(arg) == 'True':
                    save_time_rms = True
                elif str(arg) == 'False':
                    save_time_rms = False
                else:
                    raise ValueError
            elif opt == '--metric':
                metric = str(arg)
                if metric not in ['pmap_max_wass_dist', 'mean_rms', 'max_rms', 'mss_var']:
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
                if machine not in ['skynet', 'deepthought2', 'personal']:
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
        if tau_flag:
            if system == 'lorenz':
                tau = 0.1
            elif system in ['KS', 'KS_d2175']:
                tau = 0.25
    else:
        train_time, res_size, noise_realizations, save_time_rms, metric,\
                return_all, machine, rho, sigma, leakage, tau, win_type, \
                bias_type, res_per_test, num_tests, num_trains, savepred, \
                noisetype, traintype, system = argv
    if return_all and savepred:
        print('Cannot return results for all parameters and full predictions due to memory constraints.')
        raise ValueError
    if savepred:
        predflag = 'wpred_'
    else:
        predflag = ''
    if save_time_rms:
        timeflag = 'savetime_'
    else:
        timeflag = ''
    if machine == 'skynet':
        root_folder = '/h/awikner/res-noise-stabilization/'
    elif machine == 'deepthought2':
        root_folder = '/lustre/awikner1/res-noise-stabilization/'
    elif machine == 'personal':
        root_folder = 'D:/'
    # print(root_folder)

    if not return_all:
        data_folder = 'Data/%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)
        run_name = '%s_more_noisetest_%s%srho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s_metric_%s' \
             % (system,predflag, timeflag, rho, sigma, leakage, win_type, bias_type, tau, res_size, \
             train_time, noise_realizations, noisetype, traintype, metric)
    elif return_all:
        data_folder = 'Data/%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)
        run_name = '%s_%s_%srho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s' % (system,predflag, timeflag, rho, sigma, leakage, win_type, bias_type, tau,res_size, train_time, noise_realizations, noisetype, traintype)
    if runflag:
        if not os.path.isdir(os.path.join(root_folder, data_folder)):
            os.mkdir(os.path.join(root_folder, data_folder))
        if not os.path.isdir(os.path.join(os.path.join(root_folder, data_folder), run_name + '_folder')):
            os.mkdir(os.path.join(os.path.join(root_folder, data_folder), run_name + '_folder'))

        return root_folder, data_folder, run_name, system, noisetype, traintype, savepred, save_time_rms, rho,\
            sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations,\
            res_per_test, num_trains, num_tests, debug_mode, pmap, metric, return_all, ifray, machine
    else:
        if not savepred:
            return os.path.join(os.path.join(root_folder, data_folder), run_name + '.bz2'), ''
        else:
            return os.path.join(os.path.join(root_folder, data_folder), run_name + '.bz2'), \
                        os.path.join(os.path.join(root_folder, data_folder), run_name + '_folder')

