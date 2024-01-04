from ._lbp import _local_binary_pattern, _uniform_lbp_uint8
import numpy as np

def local_binary_pattern(image: np.ndarray, P: int, R: float, method: int):
    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
    assert len(image.shape) == 2
    image = np.ascontiguousarray(image, dtype=np.float64)
    return _local_binary_pattern(image, P, R, methods[method.lower()])

def uniform_lbp_uint8(image: np.ndarray, P: int, R: float):
    """Compute the uniform LBPs for the image.

    This version is adjusted for low memory usage. Input is uint8, output is uint16.
    Maximum P is 65530.

    Parameters
    ----------
    image : (M, N) uint8 array
        2D grayscale image.
    P : int < 65530
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    
    Returns
    -------
    output : (M, N) uint16 array
        LBP image.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert P < 65530
    return _uniform_lbp_uint8(image, P, R)
