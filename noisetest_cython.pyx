import numpy as np

DTYPE = np.float64

def dxdt_lorenz(double[:] x, double sigma = 10., double beta = 8/3, double rho = 28.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    cdef Py_ssize_t dim = x.size

    result = np.zeros(dim, dtype = DTYPE)
    cdef double[:] result_view = result

    result_view[0] = sigma*(x[1] - x[0])
    result_view[1] = (rho - x[2])*x[0] - x[1]
    result_view[2] = x[0]*x[1] - beta*x[2]

    return result

def rk4(double[:] x, double tau):
    # Fourth order Runge-Kutta integrator
    cdef Py_ssize_t dim = x.size

    k1 = np.zeros(dim, dtype = DTYPE)
    k1int = np.zeros(dim, dtype = DTYPE)
    k2 = np.zeros(dim, dtype = DTYPE)
    k2int = np.zeros(dim, dtype = DTYPE)
    k3 = np.zeros(dim, dtype = DTYPE)
    k3int = np.zeros(dim, dtype = DTYPE)
    k4 = np.zeros(dim, dtype = DTYPE)
    xnext = np.zeros(dim, dtype = DTYPE)

    cdef double[:] k1_view = k1
    cdef double[:] k1int_view = k1int
    cdef double[:] k2_view = k2
    cdef double[:] k2int_view = k2int
    cdef double[:] k3_view = k3
    cdef double[:] k3int_view = k3int
    cdef double[:] k4_view = k4
    cdef double[:] xnext_view = xnext

    cdef Py_ssize_t i
    
    k1 = dxdt_lorenz(x)
    for i in range(dim):
        k1int_view[i] = x[i] + k1_view[i]*tau*0.5
    k2 = dxdt_lorenz(k1int)
    for i in range(dim):
        k2int_view[i] = x[i] + k2_view[i]*tau*0.5
    k3 = dxdt_lorenz(k2int)
    for i in range(dim):
        k3int_view[i] = x[i] + k3_view[i]*tau
    k4 = dxdt_lorenz(k3int)
    for i in range(dim):
        xnext_view[i] = x[i] + tau/6*(k1_view[i]+2*k2_view[i]+2*k3_view[i]+k4_view[i])
    
    return xnext

def rungekutta(double x0 = 1, double y0 = 1,double z0 = 1, double h = 0.01, double T = 100):
   
    cdef Py_ssize_t steps = int(T/h)
    cdef Py_ssize_t num_elems = 3

    output = np.zeros((3,steps+1))
    x = np.zeros(3)
    cdef double[:,:] output_view = output
    cdef double[:] x_view = x
    output_view[0,0] = x0
    output_view[1,0] = y0
    output_view[2,0] = z0
    
    cdef Py_ssize_t i,j
    #loops from t = 0 to T 
    for i in range(steps):
        x_view = rk4(output_view[:,i],h)
        for j in range(num_elems):
            output_view[j,i+1] = x_view[j]
        
    
    return output

