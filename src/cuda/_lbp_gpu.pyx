cimport cython

from cython.parallel import prange

cdef extern from "cuda/main.h":
    void process_channel_with_lbp(unsigned int* img_data, unsigned int* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval)

def cuda_lbp(cnp.uint8_t[:, ::1] image, cnp.uint8_t[:, ::1] out, int P, cnp.float64_t R):
    cdef:
        Py_ssize_t image_width = image.shape[1]
        Py_ssize_t image_height = image.shape[0]

    process_channel_with_lbp(&image[0, 0], &out[0, 0], image_width, image_height, R, P)


