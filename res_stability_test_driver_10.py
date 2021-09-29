import os
import numpy as np
import time
noise_reals = np.arange(1,11,dtype = int)
kvals = np.arange(2, 8, dtype = int)

"""
noisetype = 'gaussian'
traintype = 'normal'

for no in noise_reals[1:]:
    testname = 'KS_%s_%s_%d' % (traintype, noisetype, no)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
    time.sleep(1)

traintype = 'rplusq'
for no in noise_reals[1:]:
    testname = 'KS_%s_%s_%d' % (traintype, noisetype, no)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
    time.sleep(1)

traintype = 'rplusq'
noisetype = 'gaussian_onestep'
no        = 10
testname  = 'KS_%s_%s_%d' % (traintype, noisetype, no)
os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
time.sleep(1)

for k in kvals:
    noisetype = 'gaussian%dstep' % k
    testname  = 'KS_%s_%s_%d' % (traintype, noisetype, no)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
    time.sleep(1)
"""
traintype = 'gradient'
noisetype = 'none'
no        = 1
"""
testname  = 'KS_%s_%s_%d' % (traintype, noisetype, no)
os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
time.sleep(1)
"""
traintype = 'gradient12'
testname  = 'KS_%s_%s_%d' % (traintype, noisetype, no)
os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T   2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
time.sleep(1)

for k in kvals[1:]:
    traintype = 'gradientk%d' % k
    testname  = 'KS_%s_%s_%d' % (traintype, noisetype, no)
    os.system('python slurm-launch.py --exp-name %s --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=%s --traintype=%s -r %d -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 24:00:00' % (testname, noisetype, traintype, no))
    time.sleep(1)
