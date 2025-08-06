cimport cython

from cython.parallel import prange

cdef extern from "cuda"