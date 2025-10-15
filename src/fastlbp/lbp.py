import numpy as np
from numpy.typing import ArrayLike

from ._lbp import (
    _local_binary_pattern,
    _uniform_lbp_uint8,
    _uniform_lbp_uint8_masked,
    _uniform_lbp_uint8_padded_absolute,
    _uniform_lbp_uint8_padded_absolute_masked,
    _uniform_lbp_uint8_padded_absolute_patch_masked,
    _uniform_lbp_uint8_patch_masked,
)


def local_binary_pattern(image: np.ndarray, P: int, R: float, method: int):
    methods = {
        "default": ord("D"),
        "ror": ord("R"),
        "uniform": ord("U"),
        "nri_uniform": ord("N"),
        "var": ord("V"),
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


def uniform_lbp_uint8_padded_absolute(
    image: np.ndarray, P: int, R: float, abs_r: int, abs_c: int, paddings_top_bottom_left_right: ArrayLike
):
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
    abs_r, abs_c : int
        Absolute pixel coordinates of the image origin.

    Returns
    -------
    output : (M, N) uint16 array
        LBP image.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert P < 65530
    assert len(paddings_top_bottom_left_right) == 4

    top, bottom, left, right = paddings_top_bottom_left_right

    return _uniform_lbp_uint8_padded_absolute(image, P, R, abs_r, abs_c, top, bottom, left, right)


def uniform_lbp_uint8_padded_absolute_masked(
    image: np.ndarray,
    mask: np.ndarray,
    P: int,
    R: float,
    abs_r: int,
    abs_c: int,
    paddings_top_bottom_left_right: ArrayLike,
):
    """Compute the uniform LBPs for the image.

    This version is adjusted for low memory usage. Input is uint8, output is uint16.
    Maximum P is 65530.

    Parameters
    ----------
    image : (M, N) uint8 array
        2D grayscale image.
    mask : (M, N) uint8 array
        2D grayscale mask. Compute pixel if its mask value is non-zero
    P : int < 65530
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    abs_r, abs_c : int
        Absolute pixel coordinates of the image origin.

    Returns
    -------
    output : (M, N) uint16 array
        LBP image.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert P < 65530
    assert len(paddings_top_bottom_left_right) == 4

    top, bottom, left, right = paddings_top_bottom_left_right

    return _uniform_lbp_uint8_padded_absolute_masked(image, mask, P, R, abs_r, abs_c, top, bottom, left, right)


def uniform_lbp_uint8_padded_absolute_patch_masked(
    image: np.ndarray,
    patch_mask: np.ndarray,
    patchsize: int,
    P: int,
    R: float,
    abs_r: int,
    abs_c: int,
    paddings_top_bottom_left_right: ArrayLike,
):
    """Compute the uniform LBPs for the image.

    This version is adjusted for low memory usage. Input is uint8, output is uint16.
    Maximum P is 65530.

    Parameters
    ----------
    image : (M, N) uint8 array
        2D grayscale image.
    mask : (M, N) uint8 array
        2D grayscale mask. Compute pixel if its mask value is non-zero
    P : int < 65530
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    abs_r, abs_c : int
        Absolute pixel coordinates of the image origin.

    Returns
    -------
    output : (M, N) uint16 array
        LBP image.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert P < 65530
    assert len(paddings_top_bottom_left_right) == 4

    # patch_mask_shape = image.shape[0] // patchsize, image.shape[1] // patchsize

    # print(f'{patch_mask.shape = } {patch_mask_shape = } DEBUG')

    assert patch_mask.dtype == np.uint8
    # assert patch_mask.shape == patch_mask_shape

    top, bottom, left, right = paddings_top_bottom_left_right

    return _uniform_lbp_uint8_padded_absolute_patch_masked(
        image, patch_mask, patchsize, P, R, abs_r, abs_c, top, bottom, left, right
    )


def uniform_lbp_uint8_masked(image: np.ndarray, mask: np.ndarray, P: int, R: float):
    """Compute the uniform LBPs for the image with mask.

    Compute only for pixels with nonzero mask value.
    This version is adjusted for low memory usage. Input is uint8, output is uint16.
    Maximum P is 65530.

    Parameters
    ----------
    image : (M, N) uint8 array
        2D grayscale image.
    mask : (M, N) uint8 array
        2D grayscale mask. Compute pixel if its mask value is non-zero
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
    assert mask.dtype == np.uint8
    assert mask.shape == image.shape
    return _uniform_lbp_uint8_masked(image, mask, P, R)


def uniform_lbp_uint8_patch_masked(image: np.ndarray, patch_mask: np.ndarray, patchsize: int, P: int, R: float):
    """Compute the uniform LBPs for the image with a patch mask.

    Compute only for pixels in the patches with a nonzero mask value.
    This version is adjusted for low memory usage. Input is uint8, output is uint16.
    Maximum P is 65530.

    Parameters
    ----------
    image : (M, N) uint8 array
        2D grayscale image.
    patch_mask : (M//patchsize, N//patchsize) uint8 array
        2D grayscale mask. Compute pixel if it is in a patch with a non-zero mask value
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

    patch_mask_shape = image.shape[0] // patchsize, image.shape[1] // patchsize
    assert patch_mask.dtype == np.uint8
    assert patch_mask.shape == patch_mask_shape
    return _uniform_lbp_uint8_patch_masked(image, patch_mask, patchsize, P, R)


def uniform_lbp_uint8_masked_stub(image: np.ndarray, P: int, R: float):
    """DEV-ONLY. Compute the uniform LBPs for the image with all-ones mask.

    Compute for all pixels, but using an algorithm with mask.
    Used to measure performance.
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

    mask = np.ones_like(image, dtype=np.uint8)
    return uniform_lbp_uint8_masked(image, mask, P, R)
