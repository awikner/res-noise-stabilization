import os

def get_windows_path(path_base):
    path = path_base.replace('/','\\')
    path = u"\\\\?\\" + path
    return path
