# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = dtw.cpp
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt, INFINITY
from libcpp.vector cimport vector
import warnings

""" Constraint is diagonal +/- constraint (e.g. constraint=2 means window_length=5
"""

ctypedef double (*metric_ptr)(double[::1] a, double[::1])

cdef inline double d_min(double a, double b, double c):
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c


cdef inline int d_argmin(double  a, double b, double c):
    if a <= b and a <= c:
        return 0
    elif b <= c:
        return 1
    else:
        return 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euclidean_distance(double[::1] a, double[::1] b):
    cdef int i
    cdef double tmp, d
    d = 0
    for i in range(a.shape[0]):
        tmp = a[i] - b[i]
        d += tmp * tmp
    return sqrt(d)


@cython.boundscheck(False)
@cython.wraparound(False)
def check_constraint(a_shape, b_shape, constraint=0, warn=True):
    cdef int min_size = abs(a_shape - b_shape) + 1
    if constraint < min_size:
        if warn:
            warnings.warn("Constraint {} too small for sequences length {} and {}; using {}".format(constraint, a_shape, b_shape, min_size))
        constraint = min_size
    return constraint


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b):
    cdef int constraint = abs(a.shape[0] - b.shape[0]) + 1
    cost_mat, cost, align_a, align_b = __dtw1d(a, b, constraint)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
def constrained_dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b,
                       constraint=0):
    constraint = check_constraint(a.shape[0],b.shape[0],constraint)
    cost_mat, cost, align_a, align_b = __dtw1d(a, b, constraint)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef __dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b,
             int constraint):
    cdef double[:, ::1] cost_mat = create_cost_mat_1d(a, b, constraint)
    align_a, align_b, cost = traceback(cost_mat)
    align_a.reverse()
    align_b.reverse()
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] create_cost_mat_1d(double[::1] a, double[::1]b, int constraint):
    cdef double[:, ::1] cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
            cost_mat[i, j] = fabs(a[i - 1] - b[j - 1]) +\
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])
    return cost_mat #[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
          np.ndarray[np.float64_t, ndim=2, mode="c"] b, metric="euclidean"):
    cdef int constraint = abs(a.shape[0] - b.shape[0]) + 1
    cost_mat, cost, align_a, align_b = __dtw2d(a, b, constraint, metric)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
def constrained_dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
                       np.ndarray[np.float64_t, ndim=2, mode="c"] b, constraint=0, metric="euclidean"):
    constraint = check_constraint(a.shape[0],b.shape[0],constraint)
    cost_mat, cost, align_a, align_b = __dtw2d(a, b, constraint, metric)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef __dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
             np.ndarray[np.float64_t, ndim=2, mode="c"] b, int constraint, metric):
    assert a.shape[1] == b.shape[1], 'Matrices must have same dimention. a={}, b={}'.format(a.shape[1], b.shape[1])
    cdef metric_ptr dist_func
    if metric == 'euclidean':
        dist_func = &euclidean_distance
    else:
        raise ValueError("unrecognized metric")
    cdef double[:, ::1] cost_mat = create_cost_mat_2d(a, b, constraint, dist_func)
    align_a, align_b, cost = traceback(cost_mat)
    align_a.reverse()
    align_b.reverse()
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] create_cost_mat_2d(double[:, ::1] a, double[:, ::1] b, int constraint, metric_ptr dist_func):
    cdef double[:, ::1] cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
            cost_mat[i, j] = dist_func(a[i - 1], b[j - 1]) +\
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])
    return cost_mat #[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] _refill_cost_matrix(double[:, ::1] a, double[:, ::1] b, double[:, ::1] cost_mat, int start_a, int end_a, int start_b, int end_b, int constraint, str metric):
    """ Refill end should include the buffer, since all of these distances need to be recalculated
    
    Args:
        a: GT sequence
        b: Pred sequence
        cost_mat: previous cost matrix
        start_a: idx of rows to start refilling
        end_a: idx of rows to end refilling
        start_b: 
        end_b: 
        constraint: 
        metric: 

    Returns:

    """
    cdef metric_ptr dist_func
    if metric == 'euclidean':
        dist_func = &euclidean_distance
    else:
        raise ValueError("unrecognized metric")

    for i in range(start_a + 1, end_a + 1): # +1 since cost mat is 1-indexed
        for j in range(max(start_b + 1, i - constraint), min(end_b + 1, i + constraint + 1)):
            cost_mat[i, j] = dist_func(a[i - 1], b[j - 1]) + \
                            d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])

    return cost_mat #[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
def refill_cost_matrix(np.ndarray[np.float64_t, ndim=2, mode="c"] a, np.ndarray[np.float64_t, ndim=2, mode="c"] b,
                       np.ndarray[np.float64_t, ndim=2, mode="c"] cost_mat, start_a, end_a, start_b, end_b, constraint, metric='euclidean'):

    #cdef np.ndarray[np.float64_t, ndim=2, mode="c"]
    #cost_mat2 = np.ascontiguousarray(cost_mat.base) # get the original matrix back with the infs in first/last row
    new_cost_mat = _refill_cost_matrix(a, b, cost_mat, start_a, end_a, start_b, end_b, constraint, metric=metric)
    return new_cost_mat


@cython.boundscheck(False)
@cython.wraparound(False)
cdef traceback(double[:, ::1] cost_mat):
    cdef int i, j
    i = cost_mat.shape[0] - 1 # go from shape to index
    j = cost_mat.shape[1] - 1
    #cost_mat = cost_mat[1:, 1:]
    cdef double cost = cost_mat[i+1, j+1] # because of inf rows/cols
    cdef vector[int] a
    cdef vector[int] b
    a.push_back(i-1) # because cost matrix is too big; corner item is i-1 idx of a
    b.push_back(j-1)
    cdef int match
    while (i > 1 or j > 1):
        match = d_argmin(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
        if match == 0:
            i -= 1
            j -= 1
        elif match == 1:
            i -= 1
        else:
            j -= 1
        a.push_back(i-1)
        b.push_back(j-1)
    return a, b, cost

@cython.boundscheck(False)
@cython.wraparound(False)
def traceback2(np.ndarray[np.float64_t, ndim=2, mode="c"] cost_mat):
    a, b, cost = traceback(cost_mat)
    a.reverse()
    b.reverse()
    return a, b, cost


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw2d_with_backward(
            np.ndarray[np.float64_t, ndim=2, mode="c"] a,
            np.ndarray[np.float64_t, ndim=2, mode="c"] b,
            np.ndarray[np.float64_t, ndim=2, mode="c"] b2,
            metric="euclidean"):
    cost_mat, cost, align_a, align_b = __dtw2d_with_backward(a, b, b2, b.shape[0], metric)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef __dtw2d_with_backward(
            np.ndarray[np.float64_t, ndim=2, mode="c"] a,
            np.ndarray[np.float64_t, ndim=2, mode="c"] b,
            np.ndarray[np.float64_t, ndim=2, mode="c"] b2,
            int constraint, metric):
    assert a.shape[1] == b.shape[1], 'Matrices must have same dimension. a={}, b={}'.format(a.shape[1], b.shape[1])
    cdef metric_ptr dist_func
    if metric == 'euclidean':
        dist_func = &euclidean_distance
    else:
        raise ValueError("unrecognized metric")
    cdef double[:, ::1] cost_mat = create_cost_mat_2d_with_backward(a, b, b2, constraint, dist_func)
    align_a, align_b, cost = traceback(cost_mat)
    align_a.reverse()
    align_b.reverse()
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] create_cost_mat_2d_with_backward(
                                            double[:, ::1] a,
                                            double[:, ::1] b,
                                            double[:, ::1] b2,
                                            int constraint, metric_ptr dist_func):
    cdef double[:, ::1] cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
            cost_mat[i, j] = min(dist_func(a[i - 1], b[j - 1]), dist_func(a[i - 1], b2[j - 1])) +\
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])
    return cost_mat #[1:, 1:]