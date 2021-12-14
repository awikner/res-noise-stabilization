import os
import numpy as np
import time
from itertools import product
#from itertools import enumerate
#rhos = np.array([0.1, 0.3, 0.5, 0.7])
rhos = np.array([0.1])
#sigmas = np.array([0.5, 1.0, 1.5, 2.0])
sigmas = np.array([0.5])
#sigmas  = np.array([0.1,0.2,0.3,0.4])
#leakages = np.array([0.5,0.625,0.75,0.875,1.0])
#leakages = np.array([0.25, 0.5, 0.75, 1.0])
leakages = np.array([0.6])
rhos_mat, sigmas_mat, leakages_mat = np.meshgrid(rhos, sigmas, leakages)
tau          = 0.25
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

#noisetypes = ['none']
#traintypes = ['gradientk%d' % k for k in np.arange(1,11)]
res_sizes_base  = [300, 400]
trainlens_base  = [5000,7000]
res_sizes, trainlens = np.meshgrid(res_sizes_base, trainlens_base)
res_sizes = res_sizes.flatten()
trainlens = trainlens.flatten()
noisetypes = ['gaussian']*trainlens.size
traintypes = ['normal']*trainlens.size
win_types  = ['full']*trainlens.size
nos        = [1]*trainlens.size
system    = 'KS'

bias_type = 'new_random'
for noisetype, res_size, win_type, traintype, no, trainlen in zip(noisetypes, res_sizes, win_types, traintypes, nos, trainlens):

    for rho, sigma, leakage in zip(rhos, sigmas, leakages):
        timestr = '%d:00:00' % ((int((trainlen-3000)/2000)+2)*2)
        print(timestr)
        testname = '%s_%s_%s_%d_%dnodes_rho%0.1f_sigma%0.1e_leakage%0.1f' % (system, traintype, noisetype, res_size, no, rho, sigma, leakage)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --savepred=False --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=25 --tests=10 --trains=25 --debug=False" --num-nodes 8 --load-env "conda activate  reservoir-rls" -t %s' % (testname, system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen, timestr))
        time.sleep(1)


