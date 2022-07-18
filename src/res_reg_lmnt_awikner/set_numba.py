import os

def set_numba(root_folder, disable_jit):
    config_file = open(os.path.join(root_folder, '.numba_config.yaml'),'w') 
    config_file.write('---\n')
    config_file.write('disable_jit: %d' % int(disable_jit))
    config_file.close() 
