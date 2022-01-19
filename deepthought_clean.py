import os
filename = '/lustre/usage/20220105/awikner1.files.txt'

with open(filename) as file:
    for line in file:
        print(line.rstrip())
        os.system('rm -rf %s' % line.rstrip())
