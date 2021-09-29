import os
with open('/lustre/usage/20210909/awikner1.files.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        print(line)
        os.system('rm -rf %s' % line)
    file.close()

