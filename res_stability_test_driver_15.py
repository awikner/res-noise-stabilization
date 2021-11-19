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
noisetypes = ['gaussian']*2
traintypes = ['normal']*2
#traintypes = ['gradientk%d' % k for k in np.arange(1,11)]
nos        = [1]*2
res_sizes  = [175, 300]
trainlens  = [4250, 2250]
win_types  = ['full']*2
system    = 'KS'

bias_type = 'new_random'

for noisetype, res_size, win_type, traintype, no, trainlen in zip(noisetypes, res_sizes, win_types, traintypes, nos, trainlens):
    for rho, sigma, leakage in zip(rhos_mat, sigmas_mat, leakages_mat):
        testname = '%s_%s_%s_%d_%dnodes_rho%0.1f_sigma%0.1e_leakage%0.1f' % (system, traintype, noisetype, res_size, no, rho, sigma, leakage)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=25 --tests=10 --trains=25 --debug=False" --num-nodes 8 --load-env "conda activate  reservoir-rls" -t 2:00:00 -A physics' % (testname, system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen))
        time.sleep(1)


