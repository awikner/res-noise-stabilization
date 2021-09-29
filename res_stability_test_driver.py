import os

noise_realizations = [1,3,6,10]
for noise in noise_realizations:
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=normal -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=normalres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=normalres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rmean -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rmeanres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rmeanres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusq -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusqres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusqres2 -r %d' % noise)

    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=normal -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=normalres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=normalres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rmean -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rmeanres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rmeanres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusq -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusqres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusqres2 -r %d' % noise)

noise = 6

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation --traintype=normal -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=normalres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=normalres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rmean -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rmeanres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rmeanres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rplusq -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rplusqres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rplusqres2 -r %d' % noise)

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=normal -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=normalres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=normalres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rmean -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rmeanres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rmeanres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rplusq -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rplusqres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rplusqres2 -r %d' % noise)

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=none --traintype=gradient -r %d' % 1)
