import os
import numpy as np
import time
noise_reals = np.arange(1,7,dtype = int)
os.system('python slurm-launch.py --exp-name test --command "python optimizingAlpha_ray_recode_KS_4.py --             noisetype=gaussian --traintype=normal -r 1 -N 1000 -T 2250" --num-nodes 4 --load-env "conda activate reservoir-rls" -t   72:00:00')
time.sleep(1)
os.system('python slurm-launch.py --exp-name test --command "python optimizingAlpha_ray_recode_KS_4.py --noisetype=gaussian --traintype=normal -r 10 -N 1000 -T 2250" --num-nodes 4 --load-env "conda activate reservoir-rls" -t 72:00:00')
time.sleep(1)
"""
for noise in noise_reals:
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian2step --traintype=rplusq -r %d -N 100 -T 50' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusq -r %d -N 100 -T 50' % noise)

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=none --traintype=gradient12 -r 1 -N 100 -T 50')
"""

