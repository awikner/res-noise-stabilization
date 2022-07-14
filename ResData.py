import pandas as pd
from tqdm import tqdm
import numpy as np
from helpers.get_run_opts import *
from climate_replication_test import *
import time
from itertools import product

class ResData:
    def __init__(self, train_time = 3000, res_size = 1000, noise_realizations = 1, \
                 save_time_rms = False, metric = 'mss_var', return_all = False, machine = 'personal', \
                 rho = 0.5, sigma = 1.0, leakage = 1.0, tau = 0.25, win_type = 'full', bias_type = 'old', \
                 res_per_test = 20, num_tests = 10, num_trains = 25, savepred = False, \
                 noisetype = 'gaussian', traintype = 'normal', system = 'KS', squarenodes = False,\
                 resonly = False, import_res = False, import_train = False, import_test = False,\
                 import_noise = False, reg_train_fracs = 1, discard_time = 500, prior = 'zero'):
        self.data_filename, self.pred_folder = get_run_opts([\
                train_time, res_size, noise_realizations, save_time_rms, metric,\
                return_all, machine, rho, sigma, leakage, tau, win_type, bias_type,\
                res_per_test, num_tests, num_trains, savepred, noisetype, traintype,\
                system, squarenodes, resonly, prior, import_res, import_train, import_test,\
                import_noise, reg_train_fracs, discard_time], runflag = False)
        print('Starding data read...')
        tic = time.perf_counter()
        print(self.data_filename)
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
        #print(sliced_data.shape)
        #print(name_list)
        #print(slice_vals)
        for name, slice_val in zip(name_list, slice_vals):
            #print(name)
            #print(slice_val)
            sliced_data = sliced_data[sliced_data[name].isin(slice_val)]
            #print(sliced_data.shape)
        #print(sliced_data.shape)
        
        if not median_flag:
            return sliced_data
        elif median_flag:
            median_slice_data = pd.DataFrame()
            remaining_vars    = [var not in reduce_axes for var in name_list[:3]]
            remaining_vars.extend([True, True, True])
            #print(remaining_vars)
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
                #print(total_vars)
                #with tqdm(total = total_vars) as pbar:
                for vars_set in product(*slice_vals[remaining_vars]):
                    #print(vars_set)
                    row_dict = {}
                    #reduced_sliced_data = sliced_data[(sliced_data.loc[:, name_list[remaining_vars]] == vars_set).all(axis = 1)]
                    reduced_sliced_data = sliced_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        reduced_sliced_data = reduced_sliced_data[reduced_sliced_data[name] == var]
                    if reduced_sliced_data.size == 0:
                        #print('No data found for ')
                        #print('Slice Vals (Res, Train, Test, Reg_Train, Noise, Reg)')
                        #print(slice_vals)
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
                        #print(row_dict)
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
                        #print(reduce_fun(pd.DataFrame(valid_times)).to_numpy()[0])
                        row_dict['valid_time'] = reduce_fun(pd.DataFrame(valid_times)).to_numpy()[0]
                        row_dict['gross_count'] = nan_count; row_dict['gross_frac'] = nan_frac
                        row_dict['data_present'] = True
                        median_slice_data = median_slice_data.append(row_dict, ignore_index = True)
                    #pbar.update(1)
            #print(median_slice_data.shape)
            if len(metric) == 0:
                return median_slice_data
            else:
                best_median_slice_data = pd.DataFrame()
                remaining_vars = [var not in reduce_axes for var in name_list]
                total_vars = 1
                for slice_val in slice_vals[remaining_vars]:
                    total_vars *= slice_val.size
                #print(total_vars)
                #with tqdm(total = total_vars) as pbar:
                for vars_set in product(*slice_vals[remaining_vars]):
                    #print(vars_set)
                    row_dict = {}
                    best_reduced_slice_data = median_slice_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data[name] == var]
                    #print(best_reduced_slice_data[metric])
                    if metric == 'valid_time':
                        #for i in range(best_reduced_slice_data[metric].size):
                        #    print(best_reduced_slice_data[metric].loc[i].to_numpy()[0])
                        #print(best_reduced_slice_data[metric])
                        #print(np.max(best_reduced_slice_data[metric]))
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
                    #pbar.update(1)
                return best_median_slice_data
            
    def data_slice_mean(self, res = np.array([]), train = np.array([]), test = np.array([]),\
              noise = np.array([]), reg = np.array([]), mean_flag = False, \
              reduce_axes = [], metric = '', gross_err_bnd = 1e2):
        input_list = [res, train, test, noise, reg]
        name_list  = np.array(['res', 'train', 'test', 'noise', 'reg'])
        data_names = [name for name in self.data.columns if name not in name_list]
        base_list  = [self.res, self.train, self.test, self.noise, self.reg]
        slice_vals = np.zeros(len(input_list), dtype = object)
        if mean_flag:
            if not isinstance(reduce_axes, list):
                print('reduce_axes must be a list.')
                return ValueError
            elif len(reduce_axes) == 0:
                print('mean_flag is True, but no axes to compute the mean over are specified.')
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
        #print(sliced_data.shape)
        
        if not mean_flag:
            return sliced_data
        elif mean_flag:
            mean_slice_data = pd.DataFrame()
            remaining_vars    = [var not in reduce_axes for var in name_list[:3]]
            remaining_vars.extend([True, True])
            
                
            
            #print(remaining_vars)
            for vars_set in product(*slice_vals[remaining_vars]):
                #print(vars_set)
                row_dict = {}
                reduced_sliced_data = sliced_data[(sliced_data.loc[:, name_list[remaining_vars]] == vars_set).all(axis = 1)]
                #for var, name in zip(vars_set, name_list[remaining_vars]):
                #    row_dict[name] = var
                #    reduced_sliced_data = reduced_sliced_data[reduced_sliced_data[name] == var]
                nans      = pd.isna(reduced_sliced_data['mean_rms'])
                near_nans = reduced_sliced_data['mean_rms'] > gross_err_bnd
                nan_count = reduced_sliced_data[nans | near_nans].shape[0]
                nan_frac  = nan_count/reduced_sliced_data.shape[0]
                for name in data_names:
                    row_dict[name] = reduced_sliced_data[~nans & ~near_nans][name].mean()
                row_dict['gross_count'] = nan_count; row_dict['gross_frac'] = nan_frac
                mean_slice_data = mean_slice_data.append(row_dict, ignore_index = True)
            #print(median_slice_data.shape)
            if len(metric) == 0:
                return mean_slice_data
            else:
                best_mean_slice_data = pd.DataFrame()
                remaining_vars = [var not in reduce_axes for var in name_list]
                for vars_set in product(*slice_vals[remaining_vars]):
                    #print(vars_set)
                    best_reduced_slice_data = mean_slice_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data[name] == var]
                    if metric == 'valid_time':
                        best_reduced_slice_data = best_reduced_slice_data[\
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].max()]
                    else:
                        best_reduced_slice_data = best_reduced_slice_data[\
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['noise'] == \
                            best_reduced_slice_data['noise'].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['reg'] == \
                            best_reduced_slice_data['reg'].min()]
                    best_mean_slice_data = best_mean_slice_data.append(best_reduced_slice_data, ignore_index = True)
                return best_mean_slice_data