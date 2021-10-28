import os
import numpy as np
import time
from itertools import product
#from itertools import enumerate
rhos = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#rhos = np.array([0.1,0.2])
sigmas = np.array([0.5,1.0,1.5,2.0])
#sigmas  = np.array([0.1,0.2,0.3,0.4])
#leakages = np.array([0.5,0.625,0.75,0.875,1.0])
leakages = np.array([0.125, 0.25, 0.375])
rhos_mat, sigmas_mat, leakages_mat = np.meshgrid(rhos, sigmas, leakages)
tau          = 0.1
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

#noisetypes = ['none', 'gaussian']
noisetypes = ['gaussian']
traintype = 'normal'
no        = 1
res_sizes  = [50, 60]
trainlen  = 500
win_types  = ['full', 'x']

"""
bias_type = 'old'

for rho, sigma, leakage in zip(rhos_mat, sigmas_mat, leakages_mat):
    testname = 'lorenz_%s_%s_%d_rho%0.1f_sigma%0.1e_leakage%0.1f' % (traintype, noisetype, no, rho, sigma, leakage)
    os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=lorenz --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --bias_type=%s --tau=%f -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t 30:00' % (testname, noisetype, traintype, no, rho, sigma, leakage, bias_type, tau, res_size, trainlen))
    #time.sleep(1)

bias_type = 'new_const'

for rho, sigma, leakage in zip(rhos_mat, sigmas_mat, leakages_mat):
     testname = 'lorenz_%s_%s_%d_rho%0.1f_sigma%0.1e_leakage%0.1f' % (traintype, noisetype, no, rho, sigma, leakage)
     os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=lorenz --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --bias_type=%s --tau=%f -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t 30:00' % (testname, noisetype, traintype, no, rho, sigma, leakage, bias_type, tau, res_size, trainlen))
"""
bias_type = 'new_random'

for noisetype, res_size, win_type in product(noisetypes, res_sizes, win_types):
    for rho, sigma, leakage in zip(rhos_mat, sigmas_mat, leakages_mat):
        testname = 'lorenz_%s_%s_%d_rho%0.1f_sigma%0.1e_leakage%0.1f' % (traintype, noisetype, no, rho, sigma, leakage)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=lorenz --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=50 --tests=20 --trains=50 --debug=False" --num-nodes 8 --load-env "conda activate  reservoir-rls" -t 30:00' % (testname, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen))

