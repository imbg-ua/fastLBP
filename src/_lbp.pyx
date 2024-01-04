#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos
# from .._shared.interpolation cimport bilinear_interpolation, round
# from .._shared.transform cimport integrate

cdef extern from "numpy/npy_math.h":
    cnp.float64_t NAN "NPY_NAN"

# from .._shared.fused_numerics cimport np_anyint as any_int

cnp.import_array()

#
# From skimage\_shared\fused_numerics.pxd
#

ctypedef fused np_ints:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_uints:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

ctypedef fused np_complexes:
    cnp.complex64_t
    cnp.complex128_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

ctypedef fused np_numeric:
    np_real_numeric
    np_complexes
    


#
# From skimage\_shared\interpolation.pxd
#

from libc.math cimport ceil, floor

# Redefine np_real_numeric to force cross type compilation
# this allows the output type to be different than the input dtype
# https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html#fused-types-and-arrays
ctypedef fused np_real_numeric_out:
    np_real_numeric

cdef inline Py_ssize_t coord_map(Py_ssize_t dim, long coord, char mode) noexcept nogil:
    """Wrap a coordinate, according to a given mode.

    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'S', 'R', 'E'}
        Whether to wrap, symmetric reflect, reflect or use the nearest
        coordinate if `coord` falls outside [0, dim).
    """
    cdef Py_ssize_t cmax = dim - 1
    if mode == b'S': # symmetric
        if coord < 0:
            coord = -coord - 1
        if coord > cmax:
            if <Py_ssize_t>(coord / dim) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % dim))
            else:
                return <Py_ssize_t>(coord % dim)
    elif mode == b'W': # wrap
        if coord < 0:
            return <Py_ssize_t>(cmax - ((-coord - 1) % dim))
        elif coord > cmax:
            return <Py_ssize_t>(coord % dim)
    elif mode == b'E': # edge
        if coord < 0:
            return 0
        elif coord > cmax:
            return cmax
    elif mode == b'R': # reflect (mirror)
        if dim == 1:
            return 0
        elif coord < 0:
            # How many times times does the coordinate wrap?
            if <Py_ssize_t>(-coord / cmax) % 2 != 0:
                return cmax - <Py_ssize_t>(-coord % cmax)
            else:
                return <Py_ssize_t>(-coord % cmax)
        elif coord > cmax:
            if <Py_ssize_t>(coord / cmax) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % cmax))
            else:
                return <Py_ssize_t>(coord % cmax)
    return coord


cdef inline np_real_numeric get_pixel2d(np_real_numeric* image,
                                        Py_ssize_t rows, Py_ssize_t cols,
                                        long r, long c, char mode,
                                        np_real_numeric cval) noexcept nogil:
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Pixel value at given position.

    """
    if mode == b'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols + c]
    else:
        return <np_real_numeric>(image[coord_map(rows, r, mode) * cols +
                                       coord_map(cols, c, mode)])


# Note: argument `sat` is an INTEGRAL IMAGE

cdef np_real_numeric integrate(np_real_numeric[:, ::1] sat,
                               Py_ssize_t r0, Py_ssize_t c0,
                               Py_ssize_t r1, Py_ssize_t c1) noexcept nogil:
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of np_real_numeric
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : np_real_numeric
        Sum over the given window.
    """
    cdef np_real_numeric S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]

    return S


#
# From skimage\_shared\interpolation.pxd
#

cdef inline Py_ssize_t round(np_floats r) noexcept nogil:
    return <Py_ssize_t>(
        (r + <np_floats>0.5) if (r > <np_floats>0.0) else (r - <np_floats>0.5)
    )

cdef inline void bilinear_interpolation(
        np_real_numeric* image, Py_ssize_t rows, Py_ssize_t cols,
        np_floats r, np_floats c, char mode, np_real_numeric cval,
        np_real_numeric_out* out) noexcept nogil:
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Interpolated value.

    """
    cdef np_floats dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = <long>ceil(r)
    maxc = <long>ceil(c)
    dr = r - minr
    dc = c - minc

    cdef cnp.float64_t top
    cdef cnp.float64_t bottom

    cdef np_real_numeric top_left = get_pixel2d(image, rows, cols, minr, minc, mode, cval)
    cdef np_real_numeric top_right = get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    cdef np_real_numeric bottom_left = get_pixel2d(image, rows, cols, maxr, minc, mode, cval)
    cdef np_real_numeric bottom_right = get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    out[0] = <np_real_numeric_out> ((1 - dr) * top + dr * bottom)


#
#
# From https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_texture.pyx#L87
#
#

cdef inline int _bit_rotate_right(int value, int length) nogil:
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))


def _local_binary_pattern(cnp.float64_t[:, ::1] image,
                          int P, cnp.float64_t R, char method=b'D'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) cnp.float64_t array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'D', 'R', 'U', 'N', 'V'}
        Method to determine the pattern.

        * 'D': 'default'
        * 'R': 'ror'
        * 'U': 'uniform'
        * 'N': 'nri_uniform'
        * 'V': 'var'

    Returns
    -------
    output : (N, M) array
        LBP image.
    """

    # texture weights
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)
    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.float64) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.float64) / P)
    cdef cnp.float64_t[::1] rp = np.round(rr, 5)
    cdef cnp.float64_t[::1] cp = np.round(cc, 5)

    # pre-allocate arrays for computation
    cdef cnp.float64_t[::1] texture = np.zeros(P, dtype=np.float64)
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef cnp.float64_t[:, ::1] output = np.zeros(output_shape, dtype=np.float64)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef cnp.float64_t lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute the variance features
    cdef cnp.float64_t sum_, var_, texture_i

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, cnp.float64_t, cnp.float64_t](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P):
                    if texture[i] - image[r, c] >= 0:
                        signed_texture[i] = 1
                    else:
                        signed_texture[i] = 0

                lbp = 0

                # if method == b'var':
                if method == b'V':
                    # Compute the variance without passing from numpy.
                    # Following the LBP paper, we're taking a biased estimate
                    # of the variance (ddof=0)
                    sum_ = 0.0
                    var_ = 0.0
                    for i in range(P):
                        texture_i = texture[i]
                        sum_ += texture_i
                        var_ += texture_i * texture_i
                    var_ = (var_ - (sum_ * sum_) / P) / P
                    if var_ != 0:
                        lbp = var_
                    else:
                        lbp = NAN
                # if method == b'uniform':
                elif method == b'U' or method == b'N':
                    # determine number of 0 - 1 changes
                    changes = 0
                    for i in range(P - 1):
                        changes += (signed_texture[i]
                                    - signed_texture[i + 1]) != 0
                    if method == b'N':
                        # Uniform local binary patterns are defined as patterns
                        # with at most 2 value changes (from 0 to 1 or from 1 to
                        # 0). Uniform patterns can be characterized by their
                        # number `n_ones` of 1.  The possible values for
                        # `n_ones` range from 0 to P.
                        #
                        # Here is an example for P = 4:
                        # n_ones=0: 0000
                        # n_ones=1: 0001, 1000, 0100, 0010
                        # n_ones=2: 0011, 1001, 1100, 0110
                        # n_ones=3: 0111, 1011, 1101, 1110
                        # n_ones=4: 1111
                        #
                        # For a pattern of size P there are 2 constant patterns
                        # corresponding to n_ones=0 and n_ones=P. For each other
                        # value of `n_ones` , i.e n_ones=[1..P-1], there are P
                        # possible patterns which are related to each other
                        # through circular permutations. The total number of
                        # uniform patterns is thus (2 + P * (P - 1)).

                        # Given any pattern (uniform or not) we must be able to
                        # associate a unique code:
                        #
                        # 1. Constant patterns patterns (with n_ones=0 and
                        # n_ones=P) and non uniform patterns are given fixed
                        # code values.
                        #
                        # 2. Other uniform patterns are indexed considering the
                        # value of n_ones, and an index called 'rot_index'
                        # representing the number of circular right shifts
                        # required to obtain the pattern starting from a
                        # reference position (corresponding to all zeros stacked
                        # on the right). This number of rotations (or circular
                        # right shifts) 'rot_index' is efficiently computed by
                        # considering the positions of the first 1 and the first
                        # 0 found in the pattern.

                        if changes <= 2:
                            # We have a uniform pattern
                            n_ones = 0  # determines the number of ones
                            first_one = -1  # position was the first one
                            first_zero = -1  # position of the first zero
                            for i in range(P):
                                if signed_texture[i]:
                                    n_ones += 1
                                    if first_one == -1:
                                        first_one = i
                                else:
                                    if first_zero == -1:
                                        first_zero = i
                            if n_ones == 0:
                                lbp = 0
                            elif n_ones == P:
                                lbp = P * (P - 1) + 1
                            else:
                                if first_one == 0:
                                    rot_index = n_ones - first_zero
                                else:
                                    rot_index = P - first_one
                                lbp = 1 + (n_ones - 1) * P + rot_index
                        else:  # changes > 2
                            lbp = P * (P - 1) + 2
                    else:  # method != 'N'
                        if changes <= 2:
                            for i in range(P):
                                lbp += signed_texture[i]
                        else:
                            lbp = P + 1
                else:
                    # method == b'default'
                    for i in range(P):
                        lbp += signed_texture[i] * weights[i]

                    # method == b'ror'
                    if method == b'R':
                        # shift LBP P times to the right and get minimum value
                        rotation_chain[0] = <int>lbp
                        for i in range(1, P):
                            rotation_chain[i] = \
                                _bit_rotate_right(rotation_chain[i - 1], P)
                        lbp = rotation_chain[0]
                        for i in range(1, P):
                            lbp = min(lbp, rotation_chain[i])

                output[r, c] = lbp

    return np.asarray(output)
