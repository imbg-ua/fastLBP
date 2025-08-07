#ifndef __LBP_CUDA_HELPERS_H__
#define __LBP_CUDA_HELPERS_H__

#include <stdio.h>

typedef struct {
	float* rr;
    float* cc;
} SampledCirclePoints;


int get_pixel2d(unsigned int* img, size_t rows, size_t cols, size_t r, size_t c, char mode, float cval);
inline void bilinear_interpolation(unsigned int* img, size_t rows, size_t cols, float r, float c, char mode, float cval, float* out);
SampledCirclePoints* sample_points_from_neighborhood(unsigned int radius, unsigned int npoints);
void sampled_points_delete(SampledCirclePoints* sampled_points);


#endif // __LBP_CUDA_HELPERS_H__