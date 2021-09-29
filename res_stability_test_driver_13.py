import os
import numpy as np
import time
#from itertools import enumerate
trainlen_0 = 50
#trainlens  = np.arange(30, 100, 5)
res_size_0 = 100
#res_sizes  = np.arange(30, 250, 10)
#base_time = 10
realizations = np.arange(1,7,dtype = int)

noisetype = 'gaussian'
traintype = 'rplusq'

times     = [2 for item in realizations]
print(times)

for i, no in enumerate(realizations):
    testname = 'lorenz_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size_0, trainlen_0)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --system=lorenz --noisetype=%s --traintype=%s -r %d -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size_0, trainlen_0, times[i]))
    time.sleep(1)

print(times)
kvals = np.arange(1,7,dtype=int)
times = [2 for k in kvals]
no = 6
for i, k in enumerate(kvals):
    if k == 1:
        noisetype = 'gaussian_onestep'
    else:
        noisetype = 'gaussian%dstep' % k
    testname = 'lorenz_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size_0, trainlen_0)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --system=lorenz --noisetype=%s --traintype=%s -r %d -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size_0, trainlen_0, times[i]))
    time.sleep(1)




