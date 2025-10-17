cimport cython
cimport numpy as cnp


cdef extern from "stdint.h":
    ctypedef unsigned char uint8_t
    ctypedef unsigned int uint32_t

cdef extern from "cuda/main.h":
    void process_channel_with_lbp(uint8_t* img_data, uint32_t* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval)

def cuda_lbp(cnp.uint8_t[:, ::1] image, cnp.uint32_t[:, ::1] out, int P, int R):
    cdef:
        int image_width = image.shape[1]
        int image_height = image.shape[0]

    process_channel_with_lbp(&image[0, 0], &out[0, 0], image_width, image_height, R, P, <char>'C', 0)


