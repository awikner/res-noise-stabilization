import pkg_resources, os
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
isray = [('ray==' in elem) for elem in installed_packages_list]
if (True in isray):
    print('Ray installed')
else:
    os.system('pip install -r -U ray')

