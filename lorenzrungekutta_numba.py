#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:26:12 2020

@author: josephharvey
"""
import numpy as np
from numba import jit
# from matplotlib import pyplot as plt

# a = 10
# b = 28
# c = 8/3
# h = 0.01
# T = 100
@jit(nopython = True, fastmath = True)
def dxdt_lorenz(x, sigma = 10., beta = 8/3, rho = 28.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    return np.array([sigma*(- x[0] + x[1]),\
                     rho*x[0] - x[1] - x[0]*x[2],\
                     x[0]*x[1]-beta*x[2]])

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
def rungekutta(x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100):
    steps = int(T/h)
    output = np.zeros((3,steps+1))
    output[0,0] = x0
    output[1,0] = y0
    output[2,0] = z0
    
    #loops from t = 0 to T 
    for i in range(0, steps):
        output[:,i+1] = rk4(output[:,i],h,dxdt_lorenz)
        
    
    return output

#u = rungekutta(1,1,1)

#time = np.array(range(0,2201))

#plt.plot(time, u[0])

#u_longterm = u[:,200:]
#time = np.array(range(0, u_longterm[0].size))
#plt.plot(u_longterm[2], u_longterm[0])

#u_sample = u[:, 0::10]
#print(u_sample.shape)
#time = np.array(range(0, u_sample[0].size))
#plt.plot(time, u_sample[0])

