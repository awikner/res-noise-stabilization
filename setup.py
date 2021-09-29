from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "compute_memview",
    ext_modules = cythonize("compute_memview.pyx", include_path = [numpy.get_include()], language_level = "3")
)
