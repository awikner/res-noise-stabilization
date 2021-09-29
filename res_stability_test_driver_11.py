import os
import numpy as np
import time
#from itertools import enumerate
trainlen_0 = 2250
trainlens  = np.arange(750, 3500, 500)
res_size_0 = 1000
res_sizes  = np.arange(800, 1600, 200)
kvals = np.arange(2, 8, dtype = int)
base_time = 10

noisetype = 'gaussian'
traintype = 'normal'
no        = 1

times     = [int(base_time/trainlen_0*trainlen)+7 for trainlen in trainlens]
print(times)

for i, trainlen in enumerate(trainlens):
    testname = 'KS_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size_0, trainlen)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N %d -T %d" --num-nodes 20 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size_0, trainlen, times[i]))
    time.sleep(1)

times     = [int(base_time/(res_size_0**2)*res_size**2)+7 for res_size in res_sizes]
print(times)

for i, res_size in enumerate(res_sizes):
    testname = 'KS_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size, trainlen_0)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N %d -T %d" --num-nodes 20 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size, trainlen_0, times[i]))
    time.sleep(1)




