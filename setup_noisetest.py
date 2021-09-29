from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "noisetest_cython",
    ext_modules = cythonize("noisetest_cython.pyx", include_path = [numpy.get_include()], language_level = "3")
)
