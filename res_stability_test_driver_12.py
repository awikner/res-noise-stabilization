import os
import numpy as np
import time
#from itertools import enumerate
trainlen_0 = 50
trainlens  = np.arange(30, 100, 5)
res_size_0 = 100
res_sizes  = np.arange(30, 250, 10)
base_time = 10

noisetype = 'gaussian'
traintype = 'normal'
no        = 1

times     = [2 for trainlen in trainlens]
print(times)

for i, trainlen in enumerate(trainlens):
    testname = 'lorenz_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size_0, trainlen)
    os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=lorenz --noisetype=%s --traintype=%s -r %d -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size_0, trainlen, times[i]))
    time.sleep(1)

times = [2 for res_size in res_sizes]
print(times)

for i, res_size in enumerate(res_sizes):
    testname = 'lorenz_%s_%s_%d_ressize_%d_trainlen_%d' % (traintype, noisetype, no, res_size, trainlen_0)
    os.system('python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py --system=lorenz --noisetype=%s --traintype=%s -r %d -N %d -T %d --res=50 --tests=20 --trains=50" --num-nodes 8 --load-env "conda activate reservoir-rls" -t %d:00:00' % (testname, noisetype, traintype, no, res_size, trainlen_0, times[i]))
    time.sleep(1)




