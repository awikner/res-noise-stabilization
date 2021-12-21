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
nos        = [2,3,4]
res_sizes  = [300]*len(nos)
trainlens  = [5000]*len(nos)
noisetypes = ['gaussian']*len(trainlens)
traintypes = ['normal']*len(trainlens)
win_types  = ['full']*len(trainlens)
system    = 'KS'

bias_type = 'new_random'
for noisetype, res_size, win_type, traintype, no, trainlen in zip(noisetypes, res_sizes, win_types, traintypes, nos, trainlens):

    for rho, sigma, leakage in zip(rhos, sigmas, leakages):
        testname = '%s_%s_%s_%d_%dnodes_rho%0.1f_sigma%0.1e_leakage%0.1f' % (system, traintype, noisetype, res_size, no, rho, sigma, leakage)
        os.system('python -u climate_replication_test.py --savepred=False --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=25 --tests=10 --trains=25 --debug=False --machine=skynet > %s.log' % (system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen, testname))

        print('Finished job: %s' % testname)
        time.sleep(1)


