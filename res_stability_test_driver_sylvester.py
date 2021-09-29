import os

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=none --traintype=sylvester -r %d' % 1)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=none --traintype=sylvester_wD -r %d' % 1)
