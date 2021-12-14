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
tau          = 0.1
rhos_mat     = rhos_mat.flatten()
sigmas_mat   = sigmas_mat.flatten()
leakages_mat = leakages_mat.flatten()

#noisetypes = ['none']
#traintypes = ['gradientk%d' % k for k in np.arange(1,11)]
bounds = np.arange(1,16)[:5]
bound_data_ratio = np.array([9.94585091, 4.93943023, 3.34270858, 2.55710817,2.11789496, 1.85370832, 1.67554074, 1.54860861, 1.44567106, 1.35926902, 1.27982677, 1.20310373, 1.13268271, 1.07554836, 1.03717013])[:5]

trainlens_base  = 1000
trainlens  = [round(bdr * trainlens_base) for bdr in bound_data_ratio]
res_sizes  = [150]*len(trainlens)
noisetypes = ['gaussian']*len(trainlens)
traintypes = ['confined%d' % bound for bound in bounds]
win_types  = ['full']*len(trainlens)
nos        = [1]*len(trainlens)
system    = 'lorenz'
timestr = '15:00'

bias_type = 'new_random'
for noisetype, res_size, win_type, traintype, no, trainlen in zip(noisetypes, res_sizes, win_types, traintypes, nos, trainlens):

    for rho, sigma, leakage in zip(rhos, sigmas, leakages):
        testname = '%s_%s_%s_%d_%dnodes_rho%0.1f_sigma%0.1e_leakage%0.1f' % (system, traintype, noisetype, res_size, no, rho, sigma, leakage)
        os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --savepred=True --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=4 --tests=5 --trains=4 --debug=True" --num-nodes 4 --load-env "conda activate  reservoir-rls" -t %s -p debug' % (testname, system, noisetype, traintype, no, rho, sigma, leakage, win_type, bias_type, tau, res_size, trainlen, timestr))
        time.sleep(1)


