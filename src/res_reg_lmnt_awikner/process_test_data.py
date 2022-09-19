#!/home/awikner1/miniconda3/envs/res39/bin/python -u
# Assume will be finished in no more than 18 hours
#SBATCH -d afterok:{{JOB_ID}}
#SBATCH -J {{JOB_NAME}}
#SBATCH --output={{LOG_NAME}}
#SBATCH -t 4:00:00
#SBATCH -A {{ACCOUNT}}
# Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=1
# Assume need 4 GB/core (4000 MB/core)
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
from itertools import product
import sys
import os
import numpy as np
import pandas as pd
import time

from res_reg_lmnt_awikner.classes import RunOpts
from res_reg_lmnt_awikner.helpers import get_windows_path, get_filename


def process_data(argv=None, run_opts=None):
    saved_flag = False

    tic = time.perf_counter()
    if not isinstance(argv, type(None)) and isinstance(run_opts, type(None)):
        run_opts = RunOpts(argv)

    raw_data_size = 0
    for ele in os.scandir(run_opts.run_folder_name):
        raw_data_size += os.stat(ele).st_size
    print('Raw data size: %0.2f kB' % (raw_data_size / 1000.))
    noise_vals = run_opts.noise_values_array
    reg_values = run_opts.reg_values
    reg_train_times = run_opts.reg_train_times

    #print("Regularization training times:")
    #print(reg_train_times)
    #print(type(reg_train_times))
    #print(reg_train_times.shape)

    rkTime = run_opts.test_time
    split = run_opts.sync_time
    num_vt_tests = (rkTime - split) // run_opts.max_valid_time

    # def get_stability_output(out_full, data_path, filename, noise_indices, train_indices, res_per_test, run_opts.num_tests, reg_values, savepred, save_time_rms, run_opts.pmap, rkTime, split, metric = 'mss_var', return_all = False):#metric='pmap_max_wass_dist'):
    # Function to process the output from all of the different reservoirs, trainning data sets, and noise values tested by find_stability.
    # If return_all is True, this simply unpacks the linear output from the find_stability loop.
    # If not, then this function returns only the results using the most optimal regularization (as defined by the metric) and using no regulariation.
    res_vals = np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test)
    train_vals = np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains)
    test_vals = np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)
    #print('Trains:')
    #print(train_vals)
    #print('Noise:')
    #print(noise_vals)
    #print('Res')
    #print(run_opts.res_per_test)
    #print('Regs:')
    #print(reg_values)

    stable_frac = np.zeros(
        (run_opts.res_per_test, train_vals.size, noise_vals.size, reg_values.size, reg_train_times.size))
    train_mean_rms = np.zeros(
        (run_opts.res_per_test, train_vals.size, noise_vals.size, reg_values.size, reg_train_times.size))
    train_max_rms = np.zeros(
        (run_opts.res_per_test, train_vals.size, noise_vals.size, reg_values.size, reg_train_times.size))
    mean_rms = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_values.size,
                         reg_train_times.size))
    max_rms = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_values.size,
                        reg_train_times.size))
    variances = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_values.size,
                          reg_train_times.size))
    valid_time = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, num_vt_tests,
                           reg_values.size, reg_train_times.size))
    if run_opts.save_time_rms:
        # mean_all = np.zeros((res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, (rkTime-split)+1, reg_values.size))
        # variances_all = np.zeros((res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, (rkTime-split)+1, reg_values.size))
        rms = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, (rkTime - split),
                        reg_values.size, reg_train_times.size))
    if run_opts.pmap:
        pmap_max_wass_dist = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size,
                                       reg_values.size, reg_train_times.size))
    if run_opts.save_eigenvals:
        eigenvals_in = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size),
                                dtype=object)

    #print(np.arange(run_opts.res_per_test, dtype=int))
    #print(np.arange(run_opts.num_trains, dtype=int))
    #print(list(enumerate(noise_vals)))
    #print(list(enumerate(reg_train_times)))
    print('Loading in raw data...')
    load_tic = time.perf_counter()
    if os.name == 'nt':
        for (i, res), (j, train), (k, noise), (l, reg_train_time) in product(
                enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test, dtype=int)),
                enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains, dtype=int)),
                enumerate(noise_vals), enumerate(reg_train_times)):
            stable_frac[i, j, k, :, l] = np.loadtxt(
                get_windows_path(
                    get_filename(run_opts.run_folder_name, 'stable_frac', res, train, noise, reg_train_time)),
                delimiter=',')
            train_mean_rms[i, j, k, :, l] = np.loadtxt(
                get_windows_path(
                    get_filename(run_opts.run_folder_name, 'train_mean_rms', res, train, noise, reg_train_time)),
                delimiter=',')
            train_max_rms[i, j, k, :, l] = np.loadtxt(
                get_windows_path(
                    get_filename(run_opts.run_folder_name, 'train_max_rms', res, train, noise, reg_train_time)),
                delimiter=',')
            mean_rms[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_windows_path(
                    get_filename(run_opts.run_folder_name, 'mean_rms', res, train, noise, reg_train_time)),
                           delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            max_rms[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_windows_path(
                    get_filename(run_opts.run_folder_name, 'max_rms', res, train, noise, reg_train_time)),
                           delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            variances[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_windows_path(
                    get_filename(run_opts.run_folder_name, 'variance', res, train, noise, reg_train_time)),
                           delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            for (m, test) in enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)):
                valid_time[i, j, m, k, :, :, l] = np.transpose(np.loadtxt(get_windows_path(
                    get_filename(run_opts.run_folder_name, 'valid_time', res, train, noise, reg_train_time,
                                 test_idx=test)),
                    delimiter=',')).reshape(
                    (num_vt_tests, reg_values.size))
            if run_opts.pmap:
                pmap_max_wass_dist[i, j, :, k, :, l] = np.transpose(np.loadtxt(
                    get_windows_path(get_filename(run_opts.run_folder_name, 'pmap_max_wass_dist', res, train, noise,
                                                  reg_train_time)),
                    delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            if run_opts.save_eigenvals:
                eigenvals_in[i, j, k, l] = np.loadtxt(
                    get_windows_path(
                        get_filename(run_opts.run_folder_name, 'gradreg_eigenvals', res, train, noise, reg_train_time)),
                    delimiter=',')
            if run_opts.save_time_rms:
                for (m, test) in enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)):
                    rms[i, j, m, k, :, :, l] = np.transpose(np.loadtxt(get_windows_path(
                        get_filename(run_opts.run_folder_name, 'rms', res, train, noise, reg_train_time,
                                     test_idx=test)),
                        delimiter=',')).reshape(
                        ((rkTime - split), reg_values.size))
        if run_opts.save_eigenvals:
            eigenvals = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size,
                                  eigenvals_in[0, 0, 0, 0].size))
            for i, j, k, l in product(np.arange(run_opts.res_per_test, dtype=int),
                                      np.arange(train_vals.size, dtype=int), np.arange(noise_vals.size, dtype=int),
                                      np.arange(reg_train_times.size, dtype=int)):
                eigenvals[i, j, k, l] = eigenvals_in[i, j, k, l]
            #print('Eigenvals shape:')
            #print(eigenvals.shape)
    else:
        for (i, res), (j, train), (k, noise), (l, reg_train_time) in product(
                enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test, dtype=int)),
                enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains, dtype=int)),
                enumerate(noise_vals), enumerate(reg_train_times)):
            stable_frac[i, j, k, :, l] = np.loadtxt(
                get_filename(run_opts.run_folder_name, 'stable_frac', res, train, noise, reg_train_time),
                delimiter=',')
            train_mean_rms[i, j, k, :, l] = np.loadtxt(
                get_filename(run_opts.run_folder_name, 'train_mean_rms', res, train, noise, reg_train_time),
                delimiter=',')
            train_max_rms[i, j, k, :, l] = np.loadtxt(
                get_filename(run_opts.run_folder_name, 'train_max_rms', res, train, noise, reg_train_time),
                delimiter=',')
            mean_rms[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_filename(run_opts.run_folder_name, 'mean_rms', res, train, noise, reg_train_time),
                delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            max_rms[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_filename(run_opts.run_folder_name, 'max_rms', res, train, noise, reg_train_time),
                delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            variances[i, j, :, k, :, l] = np.transpose(
                np.loadtxt(get_filename(run_opts.run_folder_name, 'variance', res, train, noise, reg_train_time),
                delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            for (m, test) in enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)):
                valid_time[i, j, m, k, :, :, l] = np.transpose(np.loadtxt(
                    get_filename(run_opts.run_folder_name, 'valid_time', res, train, noise, reg_train_time,
                                 test_idx=test),
                    delimiter=',')).reshape(
                    (num_vt_tests, reg_values.size))
            if run_opts.pmap:
                pmap_max_wass_dist[i, j, :, k, :, l] = np.transpose(np.loadtxt(
                    get_filename(run_opts.run_folder_name, 'pmap_max_wass_dist', res, train, noise, reg_train_time),
                    delimiter=',')).reshape((run_opts.num_tests, reg_values.size))
            if run_opts.save_eigenvals:
                eigenvals_in[i, j, k, l] = np.loadtxt(
                    get_filename(run_opts.run_folder_name, 'gradreg_eigenvals', res, train, noise, reg_train_time),
                    delimiter=',')
            if run_opts.save_time_rms:
                for (m, test) in enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)):
                    rms[i, j, m, k, :, :, l] = np.transpose(np.loadtxt(
                        get_filename(run_opts.run_folder_name, 'rms', res, train, noise, reg_train_time, test_idx=test),
                        delimiter=',')).reshape(
                        ((rkTime - split), reg_values.size))

    if run_opts.save_eigenvals:
        eigenvals = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size,
                              eigenvals_in[0, 0, 0, 0].size))
        for i, j, k, l in product(np.arange(run_opts.res_per_test, dtype=int), np.arange(train_vals.size, dtype=int),
                                  np.arange(noise_vals.size, dtype=int), np.arange(reg_train_times.size, dtype=int)):
            eigenvals[i, j, k, l] = eigenvals_in[i, j, k, l]
        #print('Eigenvals shape:')
        #print(eigenvals.shape)
    load_toc = time.perf_counter()
    print('All data loaded in %0.2f sec.' % (load_toc - load_tic))

    if run_opts.return_all:

        save_filename = run_opts.save_file_name
        if os.path.exists(save_filename):
            saved_flag = True
            print('Found data file with the same name. Loading...')
            save_tic = time.perf_counter()
            saved_data = pd.read_csv(save_filename, index_col=0)
            save_toc = time.perf_counter()
            print('Saved data loaded in %0.2f sec.' % (save_toc - save_tic))

        all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_idx, all_reg_train_idx = np.meshgrid(
            np.arange(run_opts.res_per_test, dtype=int), np.arange(train_vals.size, dtype=int),
            np.arange(run_opts.num_tests, dtype=int), np.arange(noise_vals.size, dtype=int),
            np.arange(reg_values.size, dtype=int), np.arange(reg_train_times.size, dtype=int))

        all_res_idx = all_res_idx.flatten()
        all_train_idx = all_train_idx.flatten()
        all_test_idx = all_test_idx.flatten()
        all_noise_idx = all_noise_idx.flatten()
        all_reg_idx = all_reg_idx.flatten()
        all_reg_train_idx = all_reg_train_idx.flatten()
        all_res = res_vals[all_res_idx]
        all_train = train_vals[all_train_idx]
        all_test = test_vals[all_test_idx]
        all_noise = noise_vals[all_noise_idx]
        all_reg = reg_values[all_reg_idx]
        all_reg_train = reg_train_times[all_reg_train_idx]

        data_dict = {'res': all_res,
                     'train': all_train,
                     'test': all_test,
                     'noise': all_noise,
                     'reg': all_reg,
                     'reg_train': all_reg_train}

        data_out = pd.DataFrame(data_dict)
        data_out['stable_frac'] = stable_frac[all_res_idx, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['train_mean_rms'] = train_mean_rms[
            all_res_idx, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['train_max_rms'] = train_max_rms[
            all_res_idx, all_train_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['mean_rms'] = mean_rms[
            all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['max_rms'] = max_rms[
            all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out['variance'] = variances[
            all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        data_out = pd.concat([data_out, pd.DataFrame(
            valid_time[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, :, all_reg_idx, all_reg_train_idx],
            columns=['valid_time%d' % i for i in range(num_vt_tests)])], axis=1)

        if run_opts.pmap:
            data_out['pmap_max_wass_dist'] = pmap_max_wass_dist[
                all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_idx, all_reg_train_idx]
        if run_opts.save_eigenvals:
            #print(data_out[all_test_idx == 0].shape)
            #print(data_out[all_test_idx == 0][['res', 'train', 'test', 'noise', 'reg', 'reg_train']])
            #print(eigenvals[all_res_idx, all_train_idx, all_noise_idx, all_reg_train_idx].shape)
            #print(['eig%d' % (i + 1) for i in range(eigenvals.shape[-1])])
            eigenval_idx = (all_test_idx == 0) & (all_reg_idx == 0)
            data_out.at[eigenval_idx, ['eig%d' % (i + 1) for i in range(eigenvals.shape[-1])]] = \
                eigenvals[all_res_idx[eigenval_idx], all_train_idx[eigenval_idx], all_noise_idx[eigenval_idx], \
                          all_reg_train_idx[eigenval_idx]]
        if run_opts.save_time_rms:
            # data_out = pd.concat([data_out, pd.DataFrame(mean_all[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx],\
            #        columns = ['mean_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            # print('Concatenated mean_all')
            # data_out = pd.concat([data_out, pd.DataFrame(variances_all[all_res, all_train_idx, all_test, all_noise_idx, :, all_reg_idx],\
            #        columns = ['variances_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            # print('Concatendated variances_all')
            data_out = pd.concat([data_out, pd.DataFrame(
                rms[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, :, all_reg_idx, all_reg_train_idx],
                columns=['rms%d' % (i + 1) for i in range((rkTime - split))])], axis=1)
            #print('Concatenated rms')

        if saved_flag:
            saved_cols = saved_data.columns.to_list()
            if set(saved_cols) != set(data_out.columns.to_list()):
                print('Saved Data set of the same name does not contain the same type of data.')
                print('Delete this file before running this code again.')
                raise ValueError
            data_out = pd.concat([data_out, saved_data], copy=False)
            data_out.drop_duplicates(['res', 'train', 'test', 'noise', 'reg', 'reg_train'], inplace=True)
            sort_tic = time.perf_counter()
            data_out.sort_values(['res', 'train', 'test', 'noise', 'reg', 'reg_train'], inplace=True, ignore_index=True)
            sort_toc = time.perf_counter()
            print('Data sorted in %0.2f sec.' % (sort_toc - sort_tic))
            raw_data_size = float(data_out.memory_usage().sum())

        print('Compressing and saving data...')
        save_tic = time.perf_counter()
        data_out.to_csv(save_filename)
        save_toc = time.perf_counter()
        print('Time to compress and save data: %0.2f sec.' % ((save_toc - save_tic)))

    # elif return_all and savepred:
    #    raise ValueError
    else:
        best_stable_frac = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_train_mean_rms = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_train_max_rms = np.zeros((run_opts.res_per_test, train_vals.size, noise_vals.size, reg_train_times.size))
        best_mean_rms = np.zeros(
            (run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_train_times.size))
        best_max_rms = np.zeros(
            (run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_train_times.size))
        best_variances = np.zeros(
            (run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_train_times.size))
        best_valid_time = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size,
                                    num_vt_tests, reg_train_times.size))
        if run_opts.save_time_rms:
            # best_mean_all = np.zeros((res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, (rkTime-split)+1))
            # best_variances_all = np.zeros((res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, (rkTime-split)+1))
            best_rms = np.zeros((run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size,
                                 (rkTime - split), reg_train_times.size))
        best_pmap_max_wass_dist = np.zeros(
            (run_opts.res_per_test, train_vals.size, run_opts.num_tests, noise_vals.size, reg_train_times.size))
        stable_frac_alpha = np.zeros(noise_vals.size, reg_train_times.size)
        best_j = np.zeros(noise_vals.size, reg_train_times.size)
        for (i, noise), (k, reg_train_time) in product(enumerate(noise_vals), enumerate(reg_train_times)):
            if run_opts.metric in ['mss_var', 'valid_time']:
                best_alpha_val = 0
            elif run_opts.metric in ['pmap_max_wass_dist', 'mean_rms', 'max_rms']:
                best_alpha_val = np.inf
            for j in range(reg_values.size):
                if run_opts.metric == 'mss_var':
                    metric_flag = np.mean(
                        stable_frac[:, :, i, j, k]) > best_alpha_val
                elif run_opts.metric == 'valid_time':
                    metric_flag = np.median(valid_time[:, :, :, i, :, j, k]) > best_alpha_val
                elif run_opts.metric == 'pmap_max_wass_dist':
                    # print(j)
                    # print(np.mean(run_opts.pmap_max_wass_dist[:, i, :, :, j-1]))
                    metric_flag = np.mean(
                        run_opts.pmap_max_wass_dist[:, :, :, i, j, k]) <= best_alpha_val
                elif run_opts.metric == 'mean_rms':
                    metric_flag = np.mean(mean_rms[:, :, :, i, j, k]) <= best_alpha_val
                elif run_opts.metric == 'max_rms':
                    metric_flag = np.median(max_rms[:, :, :, i, j, k]) <= best_alpha_val
                if metric_flag or (run_opts.metric in ['mss_var',
                                                       'valid_time'] and best_alpha_val == 0 and j == reg_values.size - 1) or \
                        (run_opts.metric in ['pmap_max_wass_dist', 'mean_rms', 'max_rms']
                         and np.isinf(best_alpha_val) and j == reg_values.size - 1):
                    if run_opts.metric == 'mss_var':
                        best_alpha_val = np.mean(stable_frac[:, :, i, j, k])
                    elif run_opts.metric == 'valid_time':
                        best_alpha_val = np.median(valid_time[:, :, :, i, :, j, k])
                    elif run_opts.metric == 'pmap_max_wass_dist':
                        best_alpha_val = np.mean(
                            run_opts.pmap_max_wass_dist[:, :, :, i, j, k])
                    elif run_opts.metric == 'mean_rms':
                        best_alpha_val = np.mean(mean_rms[:, :, :, i, j, k])
                    elif run_opts.metric == 'max_rms':
                        best_alpha_val = np.median(max_rms[:, :, :, i, j, k])
                    best_stable_frac[:, :, :, i, k] = -stable_frac[:, :, i, j, k]
                    best_train_mean_rms[:, :, :, i, k] = train_mean_rms[:, :, i, j, k]
                    best_train_max_rms[:, :, :, i, k] = train_max_rms[:, :, i, j, k]
                    best_variances[:, :, :, i, k] = variances[:, :, :, i, j, k]
                    best_mean_rms[:, :, :, i, k] = mean_rms[:, :, :, i, j, k]
                    best_max_rms[:, :, :, i, k] = max_rms[:, :, :, i, j, k]
                    best_valid_time[:, :, :, i, :, k] = valid_time[:, :, :, i, :, j, k]
                    best_pmap_max_wass_dist[:, i] = pmap_max_wass_dist[:, :, :, i, j, k]
                    stable_frac_alpha[i, k] = reg_values[j]
                    best_j[i, k] = int(j)
                    if run_opts.save_time_rms:
                        # best_mean_all[:,:,:,i] = mean_all[:,:,:,i,:,j]
                        # best_variances_all[:,:,:,i] = variances_all[:,:,:,i,:,j]
                        best_rms[:, :, :, i, :, k] = rms[:, :, :, i, :, j, k]
        all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_train_idx = np.meshgrid(
            np.arange(run_opts.res_per_test, dtype=int),
            np.arange(train_vals.size, dtype=int), np.arange(noise_vals.size, dtype=int),
            np.arange(run_opts.num_tests, dtype=int), np.arange(reg_train_times.size, dtype=int))

        all_res_idx = all_res_idx.flatten()
        all_train_idx = all_train_idx.flatten()
        all_test_idx = all_test_idx.flatten()
        all_noise_idx = all_noise_idx.flatten()
        all_reg_train_idx = all_reg_train_idx.flatten()
        all_res = res_vals[all_res_idx]
        all_train = train_vals[all_train_idx]
        all_test = test_vals[all_test_idx]
        all_noise = noise_vals[all_noise_idx]
        all_reg = reg_values[best_j[all_noise_idx, all_reg_train_idx]]
        all_reg_train = reg_train_times[all_reg_train_idx]

        data_dict = {'res': all_res,
                     'train': all_train,
                     'test': all_test,
                     'noise': all_noise,
                     'reg': all_reg,
                     'reg_train': all_reg_train}

        data_out = pd.DataFrame(data_dict)
        data_out['stable_frac'] = best_stable_frac[all_res_idx, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['train_mean_rms'] = best_train_mean_rms[all_res_idx, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['train_max_rms'] = best_train_max_rms[all_res_idx, all_train_idx, all_noise_idx, all_reg_train_idx]
        data_out['mean_rms'] = best_mean_rms[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_train_idx]
        data_out['max_rms'] = best_max_rms[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_train_idx]
        data_out['variance'] = best_variances[
            all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_train_idx]
        data_out = pd.concat([data_out, pd.DataFrame(
            best_valid_time[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, :, all_reg_train_idx],
            columns=['valid_time%d' % i for i in range(num_vt_tests)])], axis=1)

        if run_opts.pmap:
            data_out['pmap_max_wass_dist'] = best_pmap_max_wass_dist[
                all_res_idx, all_train_idx, all_test_idx, all_noise_idx, all_reg_train_idx]
        if run_opts.save_time_rms:
            # data_out = pd.concat([data_out, pd.DataFrame(best_mean_all[all_res, all_train_idx, all_test, all_noise_idx],\
            #    columns = ['mean_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            # print('Concatenated mean_all')
            # data_out = pd.concat([data_out, pd.DataFrame(best_variances_all[all_res, all_train_idx, all_test, all_noise_idx],\
            #         columns = ['variances_all%d' % i for i in range((rkTime-split)+1)])], axis = 1)
            # print('Concatenated variances_all')
            data_out = pd.concat([data_out, pd.DataFrame(
                best_rms[all_res_idx, all_train_idx, all_test_idx, all_noise_idx, :, all_reg_train_idx],
                columns=['rms%d' % i for i in range((rkTime - split))])], axis=1)
            print('Concatenated rms')

        print('Compressing and saving data...')
        save_tic = time.perf_counter()
        data_out.to_csv(run_opts.save_file_name)
        save_toc = time.perf_counter()
        print('Time to compress and save data: %0.2f sec.' % ((save_toc - save_tic)))

    comp_data_size = os.stat(run_opts.save_file_name).st_size

    if run_opts.savepred:
        if run_opts.return_all:
            pred_files = [get_filename(run_opts.run_folder_name, 'pred', res, train, noise, reg_train_time, reg=reg,
                                       test_idx=test, just_file=True) \
                          for res, train, test, noise, reg_train_time, reg in
                          zip(data_out['res'], data_out['train'], data_out['test'],
                              data_out['noise'], data_out['reg_train'], data_out['reg'])]
        else:
            noise_vals_set_idx, reg_train_times_set_idx = np.meshgrid(np.arange(noise_vals.size, dtype=int),
                                                                      np.arange(reg_train_times.size, dtype=int))
            noise_vals_set_idx = noise_vals_set_idx.flatten()
            reg_train_times_set_idx = reg_train_times_set_idx.flatten()
            noise_vals_set = noise_vals[noise_vals_set_idx]
            reg_train_times_set = reg_train_times[reg_train_times_set_idx]
            pred_files = ['pred_res%d_train%d_test%d_noise%e_regtrain%d_reg%e.csv' % (
                res, train, test, noise, reg_train_time, reg) \
                          for res, train, test, (noise, reg_train_time, reg) in
                          product(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test),
                                  np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains),
                                  np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests),
                                  zip(noise_vals_set, reg_train_times_set,
                                      reg_values[best_j[noise_vals_set_idx, reg_train_times_set_idx]]))]
        #print('Pred file names')
        #for file in pred_files:
        #    print(file)
        all_files = os.listdir(run_opts.run_folder_name)
        for file in all_files:
            if file not in pred_files and 'true_test' not in file:
                os.remove(os.path.join(run_opts.run_folder_name, file))
            else:
                os.rename(os.path.join(run_opts.run_folder_name, file), os.path.join(run_opts.save_folder_name, file))
        if len(os.listdir(run_opts.run_folder_name)) == 0:
            os.rmdir(run_opts.run_folder_name)
        pred_data_size = 0
        for ele in os.scandir(run_opts.save_folder_name):
            pred_data_size += os.stat(ele).st_size
        raw_data_size += pred_data_size
        comp_data_size += pred_data_size
    else:
        all_files = os.listdir(run_opts.run_folder_name)
        if run_opts.pmap:
            for file in all_files:
                if os.path.isfile(os.path.join(run_opts.run_folder_name,
                                               file)) and 'pmap_max_res' not in file and 'true_test' not in file:
                    os.remove(os.path.join(run_opts.run_folder_name, file))
                else:
                    os.rename(os.path.join(run_opts.run_folder_name, file), os.path.join(run_opts.save_folder_name, file))
        else:
            for file in all_files:
                if os.path.isfile(os.path.join(run_opts.run_folder_name, file)) and 'true_test' not in file:
                    os.remove(os.path.join(run_opts.run_folder_name, file))
                else:
                    os.rename(os.path.join(run_opts.run_folder_name, file), os.path.join(run_opts.save_folder_name, file))
        if len(os.listdir(run_opts.run_folder_name)) == 0:
            os.rmdir(run_opts.run_folder_name)
    print('Compressed data size: %0.3f kB' % (comp_data_size / 1000))
    print('Data compressed by %0.3f percent' % ((1. - comp_data_size / float(raw_data_size)) * 100))
    toc = time.perf_counter()
    print('Compressed Results Saved in %f sec.' % (toc - tic))


def main(argv):
    process_data(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
