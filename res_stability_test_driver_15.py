import os
import numpy as np
import time
from itertools import product
#from itertools import enumerate
#rhos = np.array([0.1,0.3,0.5,0.7])
rhos = np.array([0.1])
#rhos = np.array([0.1,0.2])
sigmas = np.array([1.0])
#sigmas  = np.array([0.1,0.2,0.3,0.4])
#leakages = np.array([0.5,0.625,0.75,0.875,1.0])
#leakages = np.array([0.25, 0.5, 0.75, 1.0])
leakages = np.array([0.5])
rhos_mat, sigmas_mat, leakages_mat = np.meshgrid(rhos, sigmas, leakages)
tau          = 0.25
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

#noisetypes = ['none', 'gaussian']
noisetypes = ['none']
traintype = 'gradientk2'
no        = 1
res_sizes  = [1000]
trainlen  = 9000
win_types  = ['full']
system    = 'KS'

bias_type = 'new_random'

for noisetype, res_size, win_type in product(noisetypes, res_sizes, win_types):
    for rho, sigma, leakage in zip(rhos_mat, sigmas_mat, leakages_mat):
        testname = '%s_%s_%s_%d_rho%0.1f_sigma%0.1e_leakage%0.1f' % (system, traintype, noisetype, no, rho, sigma, leakage)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=25 --tests=10 --trains=25 --debug=False" --num-nodes 12 --load-env "conda activate  reservoir-rls" -t 2:00:00' % (testname, system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen))

