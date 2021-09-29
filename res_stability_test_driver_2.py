import os

noise_realizations = [100,200,500]
for noise in noise_realizations:
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=normalres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=normalres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rmeanres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rmeanres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusqres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian --traintype=rplusqres2 -r %d' % noise)

    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=normalres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=normalres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rmeanres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rmeanres2 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusqres1 -r %d' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusqres2 -r %d' % noise)

noise = 200

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=normalres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=normalres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rmeanres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rmeanres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rplusqres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation  --traintype=rplusqres2 -r %d' % noise)

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=normalres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=normalres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rmeanres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rmeanres2 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rplusqres1 -r %d' % noise)
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=perturbation_onestep --traintype=rplusqres2 -r %d' % noise)
