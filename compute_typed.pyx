import numpy as np

DTYPE = np.intc


cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)


def compute(int[:,:] array_1, int[:,:] array_2, int a, int b, int c):
    """
    This function must implement the formula
    np.clip(array_1, 2, 10) * a + array_2 * b + c

    array_1 and array_2 are 2D.
    """
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:,:] result_view = result

    cdef int tmp

    cdef Py_ssize_t x,y

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result
