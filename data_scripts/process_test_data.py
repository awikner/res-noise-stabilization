#!/homes/awikner1/anaconda3/envs/res39/bin/python -u
# Assume will be finished in no more than 18 hours
#SBATCH -d afterok:{{JOB_ID}}
#SBATCH -J {{JOB_NAME}}
#SBATCH --output=log_files/{{JOB_NAME}}.log
#SBATCH -t 4:00:00
#SBATCH -A {{ACCOUNT}}
# Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=1
# Assume need 6 GB/core (6144 MB/core)
#SBATCH --mem-per-cpu=6144
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
import warnings
from itertools import product
import sys
sys.path.append('/lustre/awikner1/res-noise-stabilization/helpers/')
import getopt
import os

from datetime import datetime
import numpy as np
import pandas as pd
import time
import re
import subprocess

from get_run_opts import *

sys.path.append('/h/awikner/res-noise-stabilization/')

def main(argv):

    saved_flag = False

    tic = time.perf_counter()

    root_folder, top_folder, run_name, system, noisetype, traintype, savepred, save_time_rms, squarenodes, rho,\
        sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, noise_streams_per_test,\
        noise_values_array, alpha_values, res_per_test, num_trains, num_tests, debug_mode, pmap, metric, return_all,\
        ifray, machine, max_valid_time, prior, import_res, import_train, import_test, import_noise, reg_train_frac,\
        discard_time = get_run_opts(argv)

    raw_data_folder = os.path.join(os.path.join(root_folder, top_folder), run_name + '_folder')
    raw_data_size_str = str(subprocess.check_output('du -s %s' % raw_data_folder, shell = True))
    raw_data_size = int(re.search('(.*)/lustre', raw_data_size_str).group(1)[2:-2])
    print('Raw data size: %d kB' % raw_data_size)



    noise_vals      = np.loadtxt(os.path.join(raw_data_folder, 'test_noise_values.csv'), delimiter = ',', ndmin = 1)
    if not isinstance(noise_vals, np.ndarray):
        noise_vals = np.array([noise_vals])

    alpha_values    = np.loadtxt(os.path.join(raw_data_folder, 'test_alpha_values.csv'), delimiter = ',', ndmin = 1)
    if not isinstance(alpha_values, np.ndarray):
         alpha_values = np.array([alpha_values])

    time_test       = np.loadtxt(os.path.join(raw_data_folder, 'test_time_split.csv'), delimiter = ',', ndmin = 1)

    reg_train_times = np.loadtxt(os.path.join(raw_data_folder, 'reg_train_times.csv'), delimiter = ',', ndmin = 1)
    if not isinstance(reg_train_times, np.ndarray):
         reg_train_times = np.array([reg_train_times])


    print("Regularization training times:")
    print(reg_train_times)
    print(type(reg_train_times))
    print(reg_train_times.shape)

    rkTime       = int(time_test[0])
    split        = int(time_test[1])
    num_vt_tests = (rkTime-split) // max_valid_time

    #def get_stability_output(out_full, data_path, filename, noise_indices, train_indices, res_per_test, num_tests, alpha_values, savepred, save_time_rms, pmap, rkTime, split, metric = 'mss_var', return_all = False):#metric='pmap_max_wass_dist'):
    # Function to process the output from all of the different reservoirs, trainning data sets, and noise values tested by find_stability.
    # If return_all is True, this simply unpacks the linear output from the find_stability loop.
    # If not, then this function returns only the results using the most optimal regularization (as defined by the metric) and using no regulariation.
    train_vals = np.arange(num_trains)
    """
    print(train_vals)
    print(noise_vals)
    """
    tn, nt = np.meshgrid(train_vals, noise_vals)
    tn = tn.flatten()
    nt = nt.flatten()
    results = []
    print('Trains:')
    print(train_vals)
    print('Noise:')
    print(noise_vals)
    print('Res')
    print(res_per_test)
    print('Regs:')
    print(alpha_values)

    stable_frac = np.zeros((res_per_test, train_vals.size, noise_vals.size, alpha_values.size, reg_train_times.size))
    train_mean_rms = np.zeros((res_per_test, train_vals.size, noise_vals.size, alpha_values.size, reg_train_times.size))
    train_max_rms  = np.zeros((res_per_test, train_vals.size, noise_vals.size, alpha_values.size, reg_train_times.size))
    mean_rms = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, alpha_values.size, reg_train_times.size))
    max_rms = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, alpha_values.size, reg_train_times.size))
    variances = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, alpha_values.size, reg_train_times.size))
    valid_time = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, num_vt_tests, alpha_values.size, reg_train_times.size))
    if save_time_rms:
        #mean_all = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split)+1, alpha_values.size))
        #variances_all = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split)+1, alpha_values.size))
        rms = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split), alpha_values.size, reg_train_times.size))
    if pmap:
        pmap_max_wass_dist = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, alpha_values.size-1))

    print(np.arange(res_per_test, dtype = int))
    print(np.arange(num_trains, dtype = int))
    print(list(enumerate(noise_vals)))
    print(list(enumerate(reg_train_times)))
    print('Loading in raw data...')
    load_tic = time.perf_counter()
    for i,j,(k, noise),(l,reg_train_time) in product(np.arange(res_per_test, dtype = int),\
            np.arange(num_trains, dtype = int), enumerate(noise_vals), enumerate(reg_train_times)):
        stable_frac[i,j,k,:,l]    = np.loadtxt(os.path.join(raw_data_folder, 'stable_frac_res%d_train%d_noise%e_regtrain%d.csv' % (i,j,noise,reg_train_time)), delimiter = ',')
        train_mean_rms[i,j,k,:,l] = np.loadtxt(os.path.join(raw_data_folder, 'train_mean_rms_res%d_train%d_noise%e_regtrain%d.csv' % (i,j,noise,reg_train_time)), delimiter = ',')
        train_max_rms[i,j,k,:,l]  = np.loadtxt(os.path.join(raw_data_folder, 'train_max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (i,j,noise,reg_train_time)), delimiter = ',')
        mean_rms[i,j,:,k,:,l]     = np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'mean_rms_res%d_train%d_noise%e_regtrain%d.csv'% (i,j,noise,reg_train_time)), delimiter = ','))
        max_rms[i,j,:,k,:,l]      = np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'max_rms_res%d_train%d_noise%e_regtrain%d.csv' % (i,j,noise,reg_train_time)), delimiter = ','))
        variances[i,j,:,k,:,l]    = np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'variance_res%d_train%d_noise%e_regtrain%d.csv'% (i,j,noise,reg_train_time)), delimiter = ','))
        for m in range(num_tests):
            valid_time[i,j,m,k,:,:,l]    = np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'valid_time_res%d_train%d_test%d_noise%e_regtrain%d.csv' % (i,j,m,noise,reg_train_time)), delimiter = ','))
        if pmap:
            pmap_max_wass_dist[i,j,:,k,:,l] = np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'pmap_max_wass_dist_res%d_train%d_noise%e_regtrain%d.csv' % (i,j,noise,reg_train_time)), delimiter = ','))
        if save_time_rms:
            for m in list(range(num_tests)):
                #mean_all[i,j,l,k] = np.loadtxt(os.path.join(raw_data_folder, 'mean_all_res%d_train%d_test%d_noise%e.csv' % (i,j,l,noise)), delimiter = ',')
                #variances_all[i,j,l,k] = np.loadtxt(os.path.join(raw_data_folder, 'variances_all_res%d_train%d_test%d_noise%e.csv' % (i,j,l,noise)), delimiter = ',')
                rms[i,j,m,k,:,:,l] =np.transpose(np.loadtxt(os.path.join(raw_data_folder, 'rms_res%d_train%d_test%d_noise%e_regtrain%d.csv' % (i,j,m,noise,reg_train_time)), delimiter = ','))
    load_toc = time.perf_counter()
    print('All data loaded in %0.2f sec.' % (load_toc - load_tic))

    if return_all and not savepred:

        save_filename = os.path.join(os.path.join(root_folder, top_folder), run_name + '.bz2')
        if os.path.exists(save_filename):
            saved_flag = True
            print('Found data file with the same name. Loading...')
            save_tic = time.perf_counter()
            saved_data = pd.read_csv(save_filename, index_col = 0)
            save_toc = time.perf_counter()
            print('Saved data loaded in %0.2f sec.' % (save_toc - save_tic))

        all_res, all_train_idx, all_test, all_noise_idx, all_reg_idx, all_reg_train_idx = np.meshgrid(\
                np.arange(res_per_test, dtype = int), np.arange(train_vals.size, dtype = int),\
                np.arange(num_tests, dtype = int), np.arange(noise_vals.size, dtype = int),\
                np.arange(alpha_values.size, dtype = int), np.arange(reg_train_times.size, dtype = int))

        all_res       = all_res.flatten()
        all_train_idx = all_train_idx.flatten()
        all_test      = all_test.flatten()
        all_noise_idx = all_noise_idx.flatten()
        all_reg_idx   = all_reg_idx.flatten()
        all_reg_train_idx = all_reg_train_idx.flatten()
        all_train     = train_vals[all_train_idx]
        all_noise     = noise_vals[all_noise_idx]
        all_reg       = alpha_values[all_reg_idx]
        all_reg_train = reg_train_times[all_reg_train_idx]

        data_dict = {'res': all_res,
                    'train': all_train,
                    'test': all_test,
                    'noise': all_noise,
                    'reg': all_reg,
                    'reg_train': all_reg_train}

        data_out         = pd.DataFrame(data_dict)
        data_out['stable_frac'] = stable_frac[all_res, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['train_mean_rms'] = train_mean_rms[all_res, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['train_max_rms'] = train_max_rms[all_res, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['mean_rms'] = mean_rms[all_res, all_train_idx, all_test, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['max_rms'] = max_rms[all_res, all_train_idx, all_test, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['variance'] = variances[all_res, all_train_idx, all_test, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out = pd.concat([data_out, pd.DataFrame(valid_time[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx, all_reg_train_idx],\
                columns = ['valid_time%d' % i for i in range(num_vt_tests)])], axis = 1)

        if pmap:
            data_out['pmap_max_wass_dist'] = pmap_max_wass_dist[all_res, all_train_idx, all_test, all_noise_idx, all_reg_idx, all_reg_train_idx]
        if save_time_rms:
            #data_out = pd.concat([data_out, pd.DataFrame(mean_all[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx],\
            #        columns = ['mean_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            #print('Concatenated mean_all')
            #data_out = pd.concat([data_out, pd.DataFrame(variances_all[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx],\
            #        columns = ['variances_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            #print('Concatendated variances_all')
            data_out = pd.concat([data_out, pd.DataFrame(rms[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx, all_reg_train_idx],\
                    columns = ['rms%d' % (i+1) for i in range((rkTime-split))])], axis = 1)
            print('Concatenated rms')

        if saved_flag:
            saved_cols = saved_data.columns.to_list()
            if set(saved_cols) != set(data_out.columns.to_list()):
                print('Saved Data set of the same name does not contain the same type of data.')
                print('Delete this file before running this code again.')
                raise ValueError
            data_out = pd.concat([data_out, saved_data], copy = False)
            data_out.drop_duplicates(['res','train','test','noise','reg','reg_train'], inplace = True)
            sort_tic = time.perf_counter()
            data_out.sort_values(['res','train','test','noise','reg','reg_train'], inplace=True, ignore_index=True)
            sort_toc = time.perf_counter()
            print('Data sorted in %0.2f sec.' % (sort_toc - sort_tic))
            raw_data_size = float(data_out.memory_usage().sum())/1000.



        print('Compressing and saving data...')
        save_tic = time.perf_counter()
        data_out.to_csv(save_filename)
        save_toc = time.perf_counter()
        print('Time to compress and save data: %0.2f sec.' % ((save_toc-save_tic)))

    elif return_all and savepred:
        raise ValueError
    else:
        best_stable_frac = np.zeros((res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_train_mean_rms = np.zeros((res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_train_max_rms = np.zeros((res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_mean_rms = np.zeros((res_per_test, train_vals.size, num_test, noise_vals.size, reg_train_times.size))
        best_max_rms = np.zeros((res_per_test, train_vals.size, num_test, noise_vals.size, reg_train_times.size))
        best_variances = np.zeros((res_per_test, train_vals.size, num_test, noise_vals.size, reg_train_times.size))
        best_valid_time = np.zeros((res_per_test, train_vals.size, num_test, noise_vals.size, num_vt_tests, reg_train_times.size))
        if save_time_rms:
            #best_mean_all = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split)+1))
            #best_variances_all = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split)+1))
            best_rms = np.zeros((res_per_test, train_vals.size, num_tests, noise_vals.size, (rkTime-split), reg_train_times.size))
        best_pmap_max_wass_dist = np.zeros((res_per_test, train_vals.size, num_test, noise_vals.size, reg_train_times.size))
        stable_frac_alpha = np.zeros(noise_vals.size, reg_train_times.size)
        best_j = np.zeros(noise_vals.size, reg_train_times.size)
        for (i, noise), (k, reg_train_time) in product(enumerate(noise_vals), enumerate(reg_train_times)):
            if metric in ['mss_var', 'valid_time']:
                best_alpha_val = 0
            elif metric in ['pmap_max_wass_dist', 'mean_rms', 'max_rms']:
                best_alpha_val = np.inf
            for j in range(alpha_values.size):
                if metric == 'mss_var':
                    metric_flag = np.mean(
                        stable_frac[:, :, i,  j, k]) > best_alpha_val
                elif metric == 'valid_time':
                    metric_flag = np.median(valid_time[:,:,:,i,:,j,k]) > best_alpha_val
                elif metric == 'pmap_max_wass_dist':
                    #print(j)
                    #print(np.mean(pmap_max_wass_dist[:, i, :, :, j-1]))
                    metric_flag = np.mean(
                            pmap_max_wass_dist[:, :, :, i, j, k]) <= best_alpha_val
                elif metric == 'mean_rms':
                    metric_flag = np.mean(mean_rms[:,:,:,i,j,k]) <= best_alpha_val
                elif metric == 'max_rms':
                    metric_flag = np.median(max_rms[:,:,:,i,j,k]) <= best_alpha_val
                if metric_flag or (metric in ['mss_var', 'valid_time'] and best_alpha_val == 0 and j == alpha_values.size-1) or \
                        (metric in ['pmap_max_wass_dist', 'mean_rms', 'max_rms'] \
                        and np.isinf(best_alpha_val) and j == alpha_values.size-1):
                    if metric == 'mss_var':
                        best_alpha_val = np.mean(stable_frac[:, :, i, j, k])
                    elif metric == 'valid_time':
                        best_alpha_val = np.median(valid_time[:,:,:,i,:,j, k])
                    elif metric == 'pmap_max_wass_dist':
                        best_alpha_val = np.mean(
                                pmap_max_wass_dist[:,:,:,i,j,k])
                    elif metric == 'mean_rms':
                        best_alpha_val = np.mean(mean_rms[:,:,:,i,j,k])
                    elif metric == 'max_rms':
                        best_alpha_val = np.median(max_rms[:,:,:,i,j,k])
                    best_stable_frac[:,:,:,i,k] = -stable_frac[:, :, i, j,k]
                    best_train_mean_rms[:,:,:,i,k] = train_mean_rms[:, :, i, j,k]
                    best_train_max_rms[:,:,:,i,k] = train_max_rms[:, :, i, j,k]
                    best_variances[:,:,:,i,k] = variances[:, :, :, i, j,k]
                    best_mean_rms[:,:,:,i,k] = mean_rms[:, :, :, i, j,k]
                    best_max_rms[:,:,:,i,k] = max_rms[:, :, :, i, j,k]
                    best_valid_time[:,:,:,i,:,k] = valid_time[:, :, :, i, :,j,k]
                    best_pmap_max_wass_dist[:,i] = pmap_max_wass_dist[:, :, :, i, j, k]
                    stable_frac_alpha[i,k] = alpha_values[j]
                    best_j[i,k] = int(j)
                    if save_time_rms:
                        #best_mean_all[:,:,:,i] = mean_all[:,:,:,i,:,j]
                        #best_variances_all[:,:,:,i] = variances_all[:,:,:,i,:,j]
                        best_rms[:,:,:,i,:,k] = rms[:,:,:,i,:,j,k]
        all_res, all_train_idx, all_test, all_noise_idx, all_reg_train_idx = np.meshgrid(np.arange(res_per_test, dtype = int),\
                np.arange(train_vals.size, dtype = int), np.arange(noise_vals.size, dtype = int), \
                np.arange(num_tests, dtype = int), np.arange(reg_train_times.size, dtype = int))

        all_res       = all_res.flatten()
        all_train_idx = all_train_idx.flatten()
        all_test      = all_test.flatten()
        all_noise_idx = all_noise_idx.flatten()
        all_reg_train_idx = all_reg_train_idx.flatten()
        all_train     = train_vals[all_train_idx]
        all_noise     = noise_vals[all_noise_idx]
        all_reg       = alpha_values[best_j[all_noise_idx, all_reg_train_idx]]
        all_reg_train = reg_train_times[all_reg_train_idx]

        data_dict = {'res': all_res,
                     'train': all_train,
                     'test': all_test,
                     'noise': all_noise,
                     'reg': all_reg,
                     'reg_train': all_reg_train}

        data_out         = pd.DataFrame(data_dict)
        data_out['stable_frac'] = best_stable_frac[all_res, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['train_mean_rms'] = best_train_mean_rms[all_res, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['train_max_rms'] = best_train_max_rms[all_res, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['mean_rms'] = best_mean_rms[all_res, all_train_idx, all_test, all_noise_idx, all_reg_train_idx]
        data_out['max_rms'] = best_max_rms[all_res, all_train_idx, all_test, all_noise_idx, all_reg_train_idx]
        data_out['variance'] = best_variances[all_res, all_train_idx, all_test, all_noise_idx, all_reg_train_idx]
        data_out = pd.concat([data_out, pd.DataFrame(best_valid_time[all_res, all_train_idx, all_test, all_noise_idx,:, all_reg_train_idx],\
                columns = ['valid_time%d' % i for i in range(num_vt_tests)])], axis = 1)

        if pmap:
            data_out['pmap_max_wass_dist'] = best_pmap_max_wass_dist[all_res, all_train_idx, all_test, all_noise_idx, all_reg_train_idx]
        if save_time_rms:
            #data_out = pd.concat([data_out, pd.DataFrame(best_mean_all[all_res, all_train_idx, all_test, all_noise_idx],\
            #    columns = ['mean_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            #print('Concatenated mean_all')
            #data_out = pd.concat([data_out, pd.DataFrame(best_variances_all[all_res, all_train_idx, all_test, all_noise_idx],\
            #         columns = ['variances_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            #print('Concatenated variances_all')
            data_out = pd.concat([data_out, pd.DataFrame(best_rms[all_res, all_train_idx, all_test, all_noise_idx,: , all_reg_train_idx],\
                columns = ['rms%d' % i for i in range((rkTime-split))])], axis = 1)
            print('Concatenated rms')

        print('Compressing and saving data...')
        save_tic = time.perf_counter()
        data_out.to_csv(os.path.join(os.path.join(root_folder, top_folder), run_name + '.bz2'))
        save_toc = time.perf_counter()
        print('Time to compress and save data: %0.2f sec.' % ((save_toc-save_tic)))

    comp_data_size_str = str(subprocess.check_output('ls -l %s' % \
        (os.path.join(os.path.join(root_folder, top_folder), run_name + '.bz2')), shell = True))
    comp_data_size = int(re.search('dt-physics (.*) ', comp_data_size_str).group(1)[:-13])


    if savepred:
        noise_vals_set_idx, reg_train_times_set_idx = np.meshgrid(np.arange(noise_vals.size, dtype = int),\
                np.arange(reg_train_times, dtype = int))
        noise_vals_set_idx = noise_vals_set_idx.flatten()
        reg_train_times_set_idx = reg_train_times_set_idx.flatten()
        noise_vals_set = noise_vals[noise_vals_set_idx]
        reg_train_times_set = reg_train_times[reg_train_times_set_idx]
        pred_files = ['pred_res%d_train%d_test%d_noise%e_regtrain%d_reg%e.csv' % (i,j,k,noise,reg_train_time,reg) \
                         for i,j,k,(noise, reg_train_time, reg) in product(list(range(res_per_test)), list(range(num_trains)), \
                         list(range(num_tests)), zip(noise_vals_set, reg_train_times_set, \
                         alpha_values[best_j[noise_vals_set_idx, reg_train_times_set_idx]]))]
        all_files = os.listdir(raw_data_folder)
        for file in all_files:
            if file not in pred_files:
                os.system('rm %s' % file)
        pred_data_size_str = str(subprocess.check_output('du -sh %s' % raw_data_folder, shell = True))
        pred_data_size = int(re.search('(.*)/lustre', raw_data_size_str).group(1)[2:-2])
        comp_data_size += pred_data_size*1000
    else:
        os.system('find %s -type f -delete' % raw_data_folder)
        os.system('rm -rf %s' % raw_data_folder)
    print('Compressed data size: %0.3f kB' % (comp_data_size/1000))
    print('Data compressed by %0.3f percent' % ((1.-comp_data_size/(raw_data_size*1000))*100))
    toc = time.perf_counter()
    print('Compressed Results Saved in %f sec.' % (toc - tic))


if __name__ == "__main__":
    main(sys.argv[1:])

