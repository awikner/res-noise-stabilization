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
def rk4(x_in, tau, dxdt, noise = np.zeros(1, dtype = np.double)):
    # Fourth order Runge-Kutta integrator
    x = x_in + noise
    k1 = dxdt(x)
    k2 = dxdt(x + k1/2*tau)
    k3 = dxdt(x + k2/2*tau)
    k4 = dxdt(x + tau*k3)

    xnext = x + 1/6*tau*(k1+2*k2+2*k3+k4)
    return xnext

@jit(nopython = True, fastmath = True)
def rungekutta(u0 = np.array([1,1,1]), T = 100, tau = 0.1, int_step = 10, noise = np.zeros((1,1), dtype = np.double)):
    steps = T*int_step
    output = np.zeros((3,steps+int_step))
    output[:,0] = u0

    #loops from t = 0 to T
    if noise.size == 1 and noise[0,0] == 0.:
        for i in range(0, steps+int_step):
            output[:,i+1] = rk4(output[:,i],tau/int_step,dxdt_lorenz)
    else:
        for i in range(0, steps+int_step):
            output[:,i+1] = rk4(output[:,i],tau/int_step,dxdt_lorenz,noise=noise[:,i])


    return output[:,::int_step]

@jit(nopython = True, fastmath = True)
def rungekutta_pred(u0_array, tau = 0.1, int_step = 10, noise = np.zeros((1,1), dtype = np.double)):

    #loops from t = 0 to T
    if noise.size == 1 and noise[0, 0] == 0.:
        for i in range(0, int_step):
            u0_array = rk4(u0_array,tau/int_step,dxdt_lorenz_array)
    else:
        for i in range(0, int_step):
            u0_array = rk4(u0_array,tau/int_step,dxdt_lorenz_array, noise=noise[:,i])

    output = u0_array
    return output
