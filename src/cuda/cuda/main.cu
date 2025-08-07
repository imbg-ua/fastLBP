#include <cstdio>
#include "helpers.h"

__global__ void lbpKernel(unsigned int* img_data, unsigned int* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval) {
    int center_col = blockIdx.x * blockDim.x + threadIdx.x;
    int center_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
    // from https://github.com/matpetrone/LBP_Descriptor_CUDA/blob/master/src/main.cu
    if (center_col < width && center_row < height) {
        float *texture;
        int *signed_texture;
        texture = (float *) malloc(sizeof(float) * npoints);

        SampledCirclePoints* sampled_points;
        sampled_points = sample_points_from_neighborhood(radius, npoints);
        float *rr = sampled_points->rr;
        float *cc = sampled_points->cc;
        
        // get interpolated pixel values sampled on the circle
        for (size_t i = 0; i < npoints; ++i)
            bilinear_interpolation(img_data, height, width, center_row + rr[i], center_col + cc[i], 
                mode, cval, &texture[i]);

        // threshold interpolated sampled values with the central pixel intensity
        for (size_t i = 0; i < npoints; ++i)
            if (texture[i] - img_data[center_row * width + center_col] >= 0)
                signed_texture[i] = 1;
            else
                signed_texture[i] = 0;

        // get uniform LBP code
        unsigned int lbp = 0;
        
        unsigned short changes = 0;
        for (size_t i = 0; i < npoints - 1; ++i)
            changes += (short) ((signed_texture[i] - signed_texture[i + 1]) != 0);

        if (changes <= 2)
            for (size_t i = 0; i < npoints; ++i)
                lbp += signed_texture[i];
        else
            lbp = npoints + 1;

        out_feature_map[center_row * width + center_col] = lbp;

        sampled_points_delete(sampled_points);
    }

}

void* process_channel_with_lbp(unsigned int* img_data, unsigned int* out_feature_map, 
    int width, int height, int radius, int npoints, char mode, int cval) {

}