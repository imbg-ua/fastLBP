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
