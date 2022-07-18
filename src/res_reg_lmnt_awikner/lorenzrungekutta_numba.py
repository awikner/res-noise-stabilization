import numpy as np
from numba import jit
@jit(nopython = True, fastmath = True)
def dxdt_lorenz(x, sigma = 10., beta = 8/3, rho = 28.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    return np.array([sigma*(- x[0] + x[1]),\
                     rho*x[0] - x[1] - x[0]*x[2],\
                     x[0]*x[1]-beta*x[2]])

@jit(nopython = True, fastmath = True)
def dxdt_lorenz_array(x, sigma = 10., beta = 8/3, rho = 28.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    return np.stack((sigma*(- x[0] + x[1]),\
                      rho*x[0] - x[1] - x[0]*x[2],\
                      x[0]*x[1]-beta*x[2]))

@jit(nopython = True, fastmath = True)
def rk4(x, tau, dxdt):
    # Fourth order Runge-Kutta integrator

    k1 = dxdt(x)
    k2 = dxdt(x + k1/2*tau)
    k3 = dxdt(x + k2/2*tau)
    k4 = dxdt(x + tau*k3)

    xnext = x + 1/6*tau*(k1+2*k2+2*k3+k4)
    return xnext

@jit(nopython = True, fastmath = True)
def rungekutta(x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100, tau = 0.1):
    steps = T*int(tau/h)
    output = np.zeros((3,steps+1))
    output[0,0] = x0
    output[1,0] = y0
    output[2,0] = z0

    #loops from t = 0 to T
    for i in range(0, steps):
        output[:,i+1] = rk4(output[:,i],h,dxdt_lorenz)


    return output

@jit(nopython = True, fastmath = True)
def rungekutta_pred(u0_array, h = 0.01, tau = 0.1, int_steps = 10):

    #loops from t = 0 to T
    for i in range(0, int_steps):
        u0_array = rk4(u0_array,h,dxdt_lorenz_array)

    output = u0_array
    return output
