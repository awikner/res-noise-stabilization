import numpy as np
from pyfftw.interfaces import numpy_fft, cache

cache.enable()
fft = numpy_fft.fft
ifft = numpy_fft.ifft

from numba import jit, prange, objmode


class ModelParams():
    def __init__(N, d, tau, M = 16, const = 0):
        self.N = N
        self.d = d
        self.L = L
        self.tau = tau
        self.M = M
        self.const = const
        self.k = np.concatenate((np.arange(int(N/2)), np.array([0.]), np.arange(-int(N/2)+1, 0)))*2*np.pi/d
        L = (1+const)*k**2.0 - k**4.0
        self.E = np.exp(tau*L)
        self.E2 = np.exp(tau/2*L)
        r = np.exp(1j * np.pi * (np.arange(1, M+1)-0.5)/M)
        LR = tau*(np.zeros((1,M)) + L.reshape(-1,1)) + (np.zeros((N,1)) + r)
        self.Q  = tau*np.real(np.mean((np.exp(LR/2)-1)/LR, axis = 1))
        self.f1 = tau*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2.0))/(LR**3.0), axis = 1))
        self.f2 = tau*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/(LR**3.0), axis = 1))
        self.f3 = tau*np.real(np.mean((-4-3*LR-LR**2.0+np.exp(LR)*(4-LR))/(LR**3.0), axis = 1))
        self.g  = -0.5*1j*k

@jit(nopython = True, fastmath = True, parallel = True)
def mean_numba_axis1(mat):

    res = np.zeros(mat.shape[0])
    for i in prange(mat.shape[0]):
        res[i] = np.mean(mat[i])

    return res

@jit(nopython = True, fastmath = True)
def precompute_KS_params(N, d, tau, M = 16, const = 0):
    k = np.concatenate((np.arange(int(N/2)), np.array([0.]), np.arange(-int(N/2)+1, 0)))*2*np.pi/d
    L = (1+const)*k**2.0 - k**4.0
    E = np.exp(tau*L)
    E2 = np.exp(tau/2*L)
    r = np.exp(1j * np.pi * (np.arange(1, M+1)-0.5)/M)
    LR = tau*(np.zeros((1,M)) + L.reshape(-1,1)) + (np.zeros((N,1)) + r)
    Q  = tau*mean_numba_axis1(np.real((np.exp(LR/2)-1)/LR))
    f1 = tau*mean_numba_axis1(np.real((-4-LR+np.exp(LR)*(4-3*LR+LR**2.0))/(LR**3.0)))
    f2 = tau*mean_numba_axis1(np.real((2+LR+np.exp(LR)*(-2+LR))/(LR**3.0)))
    f3 = tau*mean_numba_axis1(np.real((-4-3*LR-LR**2.0+np.exp(LR)*(4-LR))/(LR**3.0)))
    g  = -0.5*1j*k

    return E, E2, Q, f1, f2, f3, g

#@jit(nopython = True, fastmath = True)
def get_fft(u):
    #with objmode(v = 'complex128[:]'):
    v = fft(u)
    return v

#@jit(nopython = True, fastmath = True)
def kursiv_forecast(v, g, E, E2, Q, f1, f2, f3):

    #with objmode(vnext = 'complex128[:]'):
    Nv = g*fft(np.real(ifft(v))**2.0)
    a  = E2*v + Q*Nv
    Na = g*fft(np.real(ifft(a))**2.0)
    b  = E2*v + Q*Na
    Nb = g*fft(np.real(ifft(b))**2.0)
    c  = E2*a + Q*(2*Nb - Nv)
    Nc = g*fft(np.real(ifft(c))**2.0)
    vnext  = E*v  + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    return vnext

#@jit(nopython = True, fastmath = True)
def real_ifft(v):
    #with objmode(u = 'double[:]'):
    u = np.real(ifft(v))
    return u

#@jit(nopython = True, fastmath = True)
def kursiv_predict(u0, tau = 0.25, N = 64, d = 22, T = 100):

    E, E2, Q, f1, f2, f3, g = precompute_KS_params(N, d, tau)
    steps = int(T/tau)

    u_arr = np.zeros((N, steps+1))
    u_arr[:,0] = u0
    v = get_fft(u_arr[:,0])
    for i in range(steps):
        v = kursiv_forecast(v, g, E, E2, Q, f1, f2, f3)
        unext = real_ifft(v)
        u_arr[:,i+1] = unext
    return u_arr

