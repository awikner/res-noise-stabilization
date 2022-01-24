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
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

#noisetypes = ['none']
#traintypes = ['gradientk%d' % k for k in np.arange(1,11)]
#traintypes = ['gradientk%d' % i for i in range(1,5)]
traintypes = []
traintypes.extend(['rplusq']*4)
print(traintypes)
taus       = [0.25]*len(traintypes)
nos        = [10]*len(taus)
res_sizes  = [200]*len(taus)
trainlens  = [4250]*len(taus)
#noisetypes = ['none']*4
noisetypes = []
noisetypes.extend(['gaussian_onestep'])
noisetypes.extend(['gaussian%dstep' % k for k in range(2,5)])
print(noisetypes)
#traintypes = ['normal']*len(taus)
win_types  = ['full']*len(taus)
system     = 'KS'

bias_type = 'new_random'
for noisetype, res_size, win_type, traintype, no, trainlen, tau in zip(noisetypes, res_sizes, win_types, traintypes, nos, trainlens, taus):

    for rho, sigma, leakage in zip(rhos, sigmas, leakages):
        testname = '%s_%s_%s_%d_%dnodes_%dtrain_rho%0.1f_sigma%0.1e_leakage%0.1f_tau%0.3f' % (system, traintype, noisetype, no, res_size, trainlen, rho, sigma, leakage, tau)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --savepred=False --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=40 --tests=10 --trains=40 --debug=False --metric=mss_var --returnall=True" --num-nodes 8 --load-env "conda activate  reservoir-rls" -t 8:00:00 -A physics' % (testname, system, noisetype, traintype, no,  rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen))
        #os.system('python -u climate_replication_test.py --savepred=False --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=25 --tests=10 --trains=25 --debug=False --machine=skynet --num_cpus=32 > %s.log' % (system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen, testname))

        time.sleep(1)


