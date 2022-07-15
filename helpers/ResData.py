import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from helpers.get_run_opts import *
from climate_replication_test import *
import time
from itertools import product
from helpers.get_windows_path import *

class ResPreds:
    def __init__(self, run_opts):
        self.data_filename, self.pred_folder = run_opts.run_file_name, run_opts.run_folder_name
        self.noise_vals = run_opts.noise_values_array; self.reg_train_vals = run_opts.reg_train_vals; self.reg_vals = run_opts.reg_values
        print('Starding data read...')
        #print(self.pred_folder)
        self.preds = np.zeros((run_opts.res_per_test, run_opts.num_trains, run_opts.num_tests, self.noise_vals.size, self.reg_train_vals.size, self.reg_vals.size), dtype = object)
        total_vals = self.preds.size
        with tqdm(total = total_vals) as pbar:
            for (i,res), (j,train), (k,test), (l,noise), (m,reg_train), (n,reg) in product(\
                    enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test)),\
                    enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains)),\
                    enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)), enumerate(self.noise_vals),\
                    enumerate(self.reg_train_vals), enumerate(self.reg_vals)):
                filename = os.path.join(self.pred_folder, 'pred_res%d_train%d_test%d_noise%e_regtrain%d_reg%e.csv' % \
                                             (res, train, test, noise, reg_train, reg))
                if os.name == 'nt' and len(filename) >= 260:
                    filename = get_windows_path(filename)
                self.preds[i,j,k,l,m,n] = np.loadtxt(filename, delimiter = ',')
                pbar.update(1)
        
class ResPmap:
    def __init__(self, run_opts):
        self.data_filename, self.pred_folder = run_opts.run_file_name, run_opts.run_folder_name
        self.noise_vals = run_opts.noise_values_array; self.reg_train_vals = run_opts.reg_train_vals; self.reg_vals = run_opts.reg_values
        print('Starding data read...')
        #print(self.pred_folder)
        self.preds = np.zeros((run_opts.res_per_test, run_opts.num_trains, run_opts.num_tests, self.noise_vals.size, self.reg_train_vals.size, self.reg_vals.size), dtype = object)
        total_vals = self.pmap_max.size
        with tqdm(total = total_vals) as pbar:
            for (i,res), (j,train), (k,test), (l,noise), (m,reg_train), (n,reg) in product(\
                enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test)),\
                enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains)),\
                enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)), enumerate(self.noise_vals),\
                enumerate(self.reg_train_vals), enumerate(self.reg_vals)):
                filename = os.path.join(self.pred_folder, 'pmap_max_res%d_train%d_test%d_noise%e_reg%e_regtrain%d.csv' %\
                                             (res, train, test, noise, reg, reg_train))
                if os.name == 'nt' and len(filename) >= 260:
                    filename = get_windows_path(filename)
                pmap_in = np.loadtxt(filename, delimiter = ',')
                pmap_max = [pmap_in[o,pmap_in[o] != 0.] for o in range(pmap_in.shape[0])]
                self.pmap_max[i,j,k,l,m,n] = pmap_max
                pbar.update(1)
        

class ResData:
    def __init__(self, run_opts):
        self.data_filename, self.pred_folder = run_opts.run_file_name, run_opts.run_folder_name
        print('Starding data read...')
        tic = time.perf_counter()
        #print(self.data_filename)
        if os.name == 'nt' and len(self.data_filename) >= 260:
            self.data_filename = get_windows_path(self.data_filename)
        self.data = pd.read_csv(self.data_filename, index_col = 0)
        toc = time.perf_counter()
        print('Data reading finished in %0.2f sec.' % (toc - tic))
        self.res   = pd.unique(self.data['res'])
        self.train = pd.unique(self.data['train'])
        self.test  = pd.unique(self.data['test'])
        self.noise = pd.unique(self.data['noise'])
        self.reg   = pd.unique(self.data['reg'])
        self.reg_train = pd.unique(self.data['reg_train'])
        self.nan   = pd.isna(self.data['variance'])
    def shape(self):
        return self.data.shape
    
    def size(self):
        return self.data.size
        
    def data_slice(self, res = np.array([]), train = np.array([]), test = np.array([]),\
              reg_train = np.array([]), noise = np.array([]), reg = np.array([]), median_flag = False, \
              reduce_axes = [], metric = '', gross_frac_metric = 'valid_time', gross_err_bnd = 1e2,\
              reduce_fun = pd.DataFrame.median):
        input_list = [res, train, test, reg_train, noise, reg]
        name_list  = np.array(['res', 'train', 'test', 'reg_train','noise', 'reg'])
        data_names = [name for name in self.data.columns if name not in name_list]
        base_list  = [self.res, self.train, self.test, self.reg_train, self.noise, self.reg]
        slice_vals = np.zeros(len(input_list), dtype = object)
        if median_flag:
            if not isinstance(reduce_axes, list):
                print('reduce_axes must be a list.')
                return ValueError
            elif len(reduce_axes) == 0:
                print('median_flag is True, but no axes to compute the median over are specified.')
                raise ValueError
            elif not all(axis in ['res','train','test','noise','reg'] for axis in reduce_axes) or \
                len(reduce_axes) != len(set(reduce_axes)):
                print('reduce_axis max only contain res, train, test, noise, or reg.')
                raise ValueError
            elif metric not in ['', 'mean_rms', 'max_rms', 'valid_time', 'pmap_max_wass_dist', 'gross_frac']:
                print('Metric not recognized.')
                raise ValueError
            elif len(metric) == 0 and any(axis in ['noise', 'reg'] for axis in reduce_axes):
                print('Cannot reduce axes with unspecified metric.')
                raise ValueError
        
        for i, item in enumerate(input_list):
            if isinstance(item, np.ndarray):
                if item.size == 0:
                    slice_vals[i] = base_list[i]
                else:
                    slice_vals[i] = item
            elif isinstance(item, list):
                if len(item) == 0:
                    slice_vals[i] = self.item
                else:
                    slice_vals[i] = np.array(item)
            else:
                slice_vals[i] = np.array([item])
        
        sliced_data = self.data
        for name, slice_val in zip(name_list, slice_vals):
            sliced_data = sliced_data[sliced_data[name].isin(slice_val)]
        
        if not median_flag:
            return sliced_data
        elif median_flag:
            median_slice_data = pd.DataFrame()
            remaining_vars    = [var not in reduce_axes for var in name_list[:3]]
            remaining_vars.extend([True, True, True])
            if np.all(remaining_vars):
                median_slice_data = sliced_data
                nans = pd.isna(median_slice_data['mean_rms'])
                near_nans = median_slice_data['mean_rms'] > gross_err_bnd
                median_slice_data['gross_count'] = np.zeros(median_slice_data.shape[0])
                median_slice_data['gross_frac'] = np.zeros(median_slice_data.shape[0])
                median_slice_data.loc[nans | near_nans, 'gross_count'] = 1.0
                median_slice_data.loc[nans | near_nans, 'gross_frac']  = 1.0
            else:
                total_vars = 1
                for slice_val in slice_vals[remaining_vars]:
                    total_vars *= slice_val.size
                for vars_set in product(*slice_vals[remaining_vars]):
                    row_dict = {}
                    reduced_sliced_data = sliced_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        reduced_sliced_data = reduced_sliced_data[reduced_sliced_data[name] == var]
                    if reduced_sliced_data.size == 0:
                        nan_frac = 1.0
                        for name in data_names:
                            if 'variance' in name:
                                row_dict[name] = 0.
                            elif 'valid_time' not in name:
                                row_dict[name] = 1e10
                        row_dict['valid_time'] = 0.
                        row_dict['gross_count'] = 0.
                        row_dict['gross_frac'] = 1.0
                        row_dict['data_present'] = False
                        median_slice_data = median_slice_data.append(row_dict, ignore_index = True)
                    else:           
                        nans      = pd.isna(reduced_sliced_data['mean_rms'])
                        near_nans = reduced_sliced_data['mean_rms'] > gross_err_bnd
                        nan_count = reduced_sliced_data[nans | near_nans].shape[0]
                        nan_frac  = nan_count/reduced_sliced_data.shape[0]
                        valid_times = np.array([])
                        for name in data_names:
                            if 'variance' in name:
                                row_dict[name] = reduce_fun(reduced_sliced_data.loc[~nans, name])
                            elif 'valid_time' not in name:
                                reduced_sliced_data.loc[nans, name] = 1e10
                                row_dict[name] = reduce_fun(reduced_sliced_data[name])
                            else:
                                reduced_sliced_data.loc[pd.isna(reduced_sliced_data[name]), name] = 0.0
                                valid_times = np.append(valid_times, reduced_sliced_data[name].to_numpy())
                        row_dict['valid_time'] = reduce_fun(pd.DataFrame(valid_times)).to_numpy()[0]
                        row_dict['gross_count'] = nan_count; row_dict['gross_frac'] = nan_frac
                        row_dict['data_present'] = True
                        median_slice_data = median_slice_data.append(row_dict, ignore_index = True)
            if len(metric) == 0:
                return median_slice_data
            else:
                best_median_slice_data = pd.DataFrame()
                remaining_vars = [var not in reduce_axes for var in name_list]
                total_vars = 1
                for slice_val in slice_vals[remaining_vars]:
                    total_vars *= slice_val.size
                for vars_set in product(*slice_vals[remaining_vars]):
                    row_dict = {}
                    best_reduced_slice_data = median_slice_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data[name] == var]
                    if metric == 'valid_time':
                        best_reduced_slice_data = best_reduced_slice_data[\
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].max()]
                    elif metric == 'gross_frac':
                        best_reduced_slice_data = best_reduced_slice_data[\
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].min()]
                        if len(best_reduced_slice_data.shape) != 1:
                            if gross_frac_metric == 'valid_time':
                                best_reduced_slice_data = best_reduced_slice_data[\
                                    best_reduced_slice_data[gross_frac_metric] == best_reduced_slice_data[gross_frac_metric].max()]
                            else:
                                best_reduced_slice_data = best_reduced_slice_data[\
                                    best_reduced_slice_data[gross_frac_metric] == best_reduced_slice_data[gross_frac_metric].min()]
                    else:
                        best_reduced_slice_data = best_reduced_slice_data[\
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['noise'] == \
                            best_reduced_slice_data['noise'].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['reg'] == \
                            best_reduced_slice_data['reg'].min()]
                    best_median_slice_data = best_median_slice_data.append(best_reduced_slice_data, ignore_index = True)
                return best_median_slice_data
