# Installation
Begin by creating a virtual environment with Python 3.9, preferably using Conda.
```
conda create --name res39 python=3.9
```
Then, activate this environment and install the package from TestPyPI.
```
pip3 install --extra-index-url https://test.pypi.org/simple/ res-reg-lmnt-awikner
```
Test scripts to recreate results found in the paper can be found in [res\_test\_drivers](https://github.com/awikner/res-noise-stabilization/tree/master/res_test_drivers).
