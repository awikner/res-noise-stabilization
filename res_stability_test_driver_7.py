import os
import numpy as np
import time
noise_reals = np.arange(1,7,dtype = int)
kvals = np.arange(3, 11, dtype = int)
for k in kvals:
    os.system('python slurm-launch.py --exp-name test --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=gaussian%dstep --traintype=rplusq -r 10 -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 8:00:00' % k)
    time.sleep(1)
"""
os.system('python slurm-launch.py --exp-name test --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=gaussian --traintype=normal -r 1 -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 12:00:00' % k)
time.sleep(1)

os.system('python slurm-launch.py --exp-name test --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=gaussian --traintype=rplusq -r 10 -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 16:00:00' % k)
time.sleep(1)

os.system('python slurm-launch.py --exp-name test --command "python -u optimizingAlpha_ray_recode_KS_3.py --noisetype=gaussian --traintype=normal -r 1 -N 1000 -T 2250" --num-nodes 20 --load-env "conda activate reservoir-rls" -t 12:00:00' % k)

for noise in noise_reals:
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian2step --traintype=rplusq -r %d -N 100 -T 50' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusq -r %d -N 100 -T 50' % noise)

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=none --traintype=gradient12 -r 1 -N 100 -T 50')
"""

