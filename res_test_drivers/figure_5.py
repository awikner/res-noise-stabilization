import sys,os
from res_reg_lmnt_awikner.set_numba import set_numba
from res_reg_lmnt_awikner.reservoir_train_test import start_reservoir_test
from res_reg_lmnt_awikner.process_test_data import process_data
from res_reg_lmnt_awikner.RunOpts import RunOpts
from res_reg_lmnt_awikner.ResData import ResData, ResPreds
from res_reg_lmnt_awikner.get_windows_path import get_windows_path

import subprocess
import numpy as np
import time
from itertools import product
import re
rho = 0.6
sigma = 0.1
leakage = 1.0

discard_time = 500
#traintypes   = ['normal','gradientk1','normal','gradientk1','normal','gradientk4','regzerok4']
traintypes = ['normal','gradientk1','normal','gradientk4','regzerok4']
train_time   = 20000
res_size     = 500
noise_realizations = 1
#noisetypes   = ['none','none','none','none','gaussian','none','none']
noisetypes   = ['none','none','gaussian','none','none']
tau          = 0.25
win_type     = 'full_0centered'
squarenodes  = True
system       = 'KS'
bias_type    = 'new_random'
"""
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
reg_train_times_all = [np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([4])]
"""
noise_values_array_all = [np.array([0.0]),\
                        np.array([10.**(-5.4)]),\
                        np.array([10.**(-7.4)]),\
                        np.array([10.**(-7.4)]),\
                        np.array([10.**(-7.4)])]
reg_values_all         = [np.array([10.**(-6.0)]),\
                        np.array([10.**(-8.5)]),\
                        np.array([10.**(-16)]),\
                        np.array([10.**(-16)]),\
                        np.array([10.**(-16.5)])]
reg_train_times_all = [np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([20000]),\
                        np.array([4])]

res_start = 2; train_start = 9; test_start = 2;

test_time       = 18000
return_all      = True
savepred        = True
save_time_rms   = False
debug_mode      = False
ifray           = False
just_process    = False
just_display    = False
nojit           = False
res_per_test    = 1
#res_per_test    = 4
num_trains      = 1
#num_trains      = 3
num_tests       = 1
cpus_per_node   = 1
metric          = 'mss_var'
machine         = 'personal'
max_valid_time  = 2000
prior           = 'zero'
save_eigenvals  = False
pmap            = False
set_numba(os.getcwd(),nojit)
all_data     = []
all_run_opts = []
all_preds    = []
for i, (noisetype, traintype, noise_values_array, reg_values, reg_train_times_in) in enumerate(zip(noisetypes, traintypes, noise_values_array_all, reg_values_all, reg_train_times_all)):
    run_opts = RunOpts(system = system, traintype = traintype, noisetype = noisetype, noise_realizations = noise_realizations,\
            res_size = res_size, train_time = train_time, test_time = test_time, rho = rho, sigma = sigma, leakage = leakage,\
            tau = tau, win_type = win_type, bias_type = bias_type, noise_values_array = noise_values_array,\
            savepred = savepred, save_time_rms = save_time_rms, squarenodes = squarenodes, debug_mode = debug_mode,\
            res_per_test = res_per_test, num_trains = num_trains, num_tests = num_tests,\
            metric = metric, pmap = pmap, return_all = return_all, machine = machine,\
            max_valid_time = max_valid_time, ifray = ifray,\
            reg_values = reg_values, reg_train_times = reg_train_times_in, discard_time = discard_time,\
            prior = prior, save_eigenvals = save_eigenvals, num_cpus = cpus_per_node,\
            res_start = res_start, train_start = train_start, test_start = test_start)
    if i == 0:
        run_opts.save_truth = True
    else:
        run_opts.save_truth = False
    if not just_process:
        start_reservoir_test(run_opts=run_opts)
        time.sleep(5)
    if not just_display:
        process_data(run_opts=run_opts)
    all_data.append(ResData(run_opts))
    all_preds.append(ResPreds(run_opts))
    all_run_opts.append(run_opts)

lyapunov_time = 1./tau/0.048
gross_err_bnd = 1.0

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt

plasma = cm.get_cmap('bwr', 256)
new_cmp_vals = plasma(np.linspace(0,1,256))
black = np.array([0,0,0,1]).reshape(1,-1)
new_cmp_vals = np.concatenate((np.ones((64,1)) @ black, new_cmp_vals, np.ones((64,1)) @ black), axis = 0)
print(new_cmp_vals.shape)
new_cmp = ListedColormap(new_cmp_vals)

plotlen_s = (round(7*lyapunov_time)+2)/lyapunov_time
plotlen = round(7*lyapunov_time)+2
line_width = 2
if os.name == 'nt':
    ks_true = np.loadtxt(get_windows_path(os.path.join(all_run_opts[0].run_folder_name, '%s_tau%0.2f_true_test_%d.csv' %\
        (all_run_opts[0].system, all_run_opts[0].tau, all_run_opts[0].test_start))), delimiter = ',')
else:
    ks_true = np.loadtxt(os.path.join(all_run_opts[0].run_folder_name, '%s_tau%0.2f_true_test_%d.csv' %\
                    (all_run_opts[0].system, all_run_opts[0].tau, all_run_opts[0].test_start)), delimiter = ',')
num_vars = 64
ub, lb = 4.5, -4.5
X,Y = np.meshgrid(22/64*np.arange(num_vars), (1/lyapunov_time)*np.arange(plotlen))
plt.rcParams.update({'font.size': 30})
raw_data_map   = [0,1,2]
plot_idxs = [0,1,2]
plot_type = 'stable'
xmax = plotlen_s
for j,i in enumerate(plot_idxs):
    fig = plt.figure(figsize = (17,5))
    pred_plot = np.copy(all_preds[i].preds[0,0,0,0,0,0][:, :plotlen])
    pred_plot[np.isnan(pred_plot)] = 1000.
    pred_plot[np.isinf(pred_plot)] = 1000.
    cs=plt.pcolormesh(Y.T, X.T,pred_plot,cmap = new_cmp)
    cs.cmap.set_under('k')
    cs.cmap.set_over('k')
    plt.yticks([0,22/64*np.arange(num_vars)[-1]/2,22/64*np.arange(num_vars)[-1]],['0','L/2','L'])
    plt.clim(lb,ub)
    ticks = np.append(-3.75, np.append(np.arange(-3,4), 3.75))
    tick_labels = []
    for l in range(9):
        if l == 0:
            tick_labels.append('< -3')
        elif l == 8:
            tick_labels.append('> 3')
        else:
            tick_labels.append('%d' % (l-4))
    cbar = plt.colorbar(ticks = ticks)
    cbar.ax.set_yticklabels(tick_labels)
    plt.xlim(0,xmax)
    plt.ylim(0,np.max(X))
    plt.xlabel('Lyapunov Time')
    plt.ylabel('x')
    cbar.set_label('u(t)')
    ax = fig.axes[0]
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(line_width)
    cbar.outline.set_linewidth(line_width)
    ax.tick_params(width=line_width)
    cbar.ax.tick_params(width=line_width)
    plt.savefig(os.path.join(os.getcwd(), 'KS_pred_noisetype_%s_traintype_%s_reg%e_%s.pdf' % (all_run_opts[i].noisetype,\
        all_run_opts[i].traintype, all_run_opts[i].reg_values[0], plot_type)), bbox_inches = "tight", dpi = 400)
    plt.show()


    fig = plt.figure(figsize = (17,5))
    pred_plot[np.isnan(pred_plot)] = 1000.
    pred_plot[np.isinf(pred_plot)] = 1000.
    cs=plt.pcolormesh(Y.T, X.T,pred_plot - ks_true[:,:plotlen], cmap = new_cmp)
    data_slice = all_data[raw_data_map[i]].data_slice(res=all_run_opts[i].res_start,train=all_run_opts[i].train_start,\
                                            test=all_run_opts[i].test_start,noise=all_run_opts[i].noise_values_array[0],\
                                            reg = all_run_opts[i].reg_values[0], \
                                            reg_train=all_run_opts[i].reg_train_times[0])
    vt = data_slice['valid_time0'].to_numpy()[0]/lyapunov_time
    print(vt)
    plt.plot([vt,vt],[-10,100], 'c', linewidth = 4)
    if j > 4:
        text_x = vt-0.817
    elif j < 3:
        text_x = vt+0.817
    else:
        text_x = vt+0.817
    plt.text(text_x, 2, 'VT = %0.3f' % vt,fontsize=26,ha="center", va="center",\
             bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=2,alpha=0.8))
    cs.cmap.set_under('k')
    cs.cmap.set_over('k')
    plt.yticks([0,22/64*np.arange(num_vars)[-1]/2,22/64*np.arange(num_vars)[-1]],['0','L/2','L'])
    plt.clim(lb,ub)
    ticks = np.append(-3.75, np.append(np.arange(-3,4), 3.75))
    tick_labels = []
    for l in range(9):
        if l == 0:
            tick_labels.append('< -3')
        elif l == 8:
            tick_labels.append('> 3')
        else:
            tick_labels.append('%d' % (l-4))
    cbar = plt.colorbar(ticks = ticks)
    cbar.ax.set_yticklabels(tick_labels)
    plt.xlim(0,xmax)
    plt.ylim(0,np.max(X))
    plt.xlabel('Lyapunov Time')
    plt.ylabel('x')
    cbar.set_label('u(t)')
    ax =fig.axes[0]
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(line_width)
    cbar.outline.set_linewidth(line_width)
    ax.tick_params(width=line_width)
    cbar.ax.tick_params(width=line_width)
    plt.savefig(os.path.join(os.getcwd(), 'KS_error_noisetype_%s_traintype_%s_reg%e_%s.pdf' % (all_run_opts[i].noisetype,\
            all_run_opts[i].traintype, all_run_opts[i].reg_values[0], plot_type)), bbox_inches = "tight", dpi = 400)
    plt.show()

fig = plt.figure(figsize = (17,5))
pred_plot = ks_true[:,:plotlen]
cs=plt.pcolormesh(Y.T, X.T,pred_plot,cmap = 'bwr')
cs.cmap.set_under('k')
cs.cmap.set_over('k')
plt.clim(-3,3)
cbar = plt.colorbar(ticks = np.arange(-3,4))
plt.xlim(0,np.max(Y))
plt.ylim(0,np.max(X))
plt.xlabel('Lyapunov Time')
plt.ylabel('x')
cbar.set_label('u(t)')
ax =fig.axes[0]
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(line_width)
cbar.outline.set_linewidth(line_width)
ax.tick_params(width=line_width)
cbar.ax.tick_params(width=line_width)
plt.yticks([0,22/64*np.arange(num_vars)[-1]/2,22/64*np.arange(num_vars)[-1]],['0','L/2','L'])

plt.savefig(os.path.join(os.getcwd(),'KS_truth_stable.pdf'), bbox_inches = "tight", dpi = 400)
plt.show()
