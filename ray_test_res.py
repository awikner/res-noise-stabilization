import os

os.system("python -u climate_replication_test.py --system=KS --noisetype=gaussian --savepred=False --traintype=normal -r 1 --win_type=full --rho=0.1 --sigma=0.5 --leakage=0.3 --bias_type=new_random --tau=0.125 -N 200 -T 4000 --res=5 --tests=5 --trains=4 --debug=False --machine=skynet --num_cpus=30 > KS_quicktest_4.log &")
