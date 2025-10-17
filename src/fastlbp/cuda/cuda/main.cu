#include <cstdio>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdint.h>
#include "common.h"

// const unsigned int THREADS_PER_BLOCK = 1;

#define TILE_WIDTH 16   // 16x16 = 256 threads per block (safe)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// static void CudaCheck(cudaError_t error, const char *file, int line) {
//     if (error != cudaSuccess)
//     {
//         f// printf(stderr, "Error: %s:%d, ", file, line);
//         f// printf(stderr, "code: %d, reason: %s\n", error,
//                 cudaGetErrorString(error));
//         exit( EXIT_FAILURE );
//     }
// }


#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))

typedef struct {
	float* rr;
    float* cc;
} SampledCirclePoints;

__device__ int get_pixel2d(uint8_t* img, size_t rows, size_t cols, int r, int c, char mode, float cval) {

    // only constant mode is implemented
    if (mode == 'C') {
        if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) {
            return cval;
        } else 
            return img[r * cols + c];
    }
    else {
        // other modes are not supported
        return 0;
    }

    return 0;
    
}

__device__ inline void bilinear_interpolation(uint8_t* image, size_t rows, size_t cols, float r, float c, char mode, float cval, float* out) {
    float dr, dc;
    long minr, minc, maxr, maxc;

    minr = (long)floor(r);
    minc = (long)floor(c);
    maxr = (long)ceil(r);
    maxc = (long)ceil(c);
    
    dr = r - minr;
    dc = c - minc;

    float top;
    float bottom;

    float top_left = get_pixel2d(image, rows, cols, minr, minc, mode, cval);
    float top_right = get_pixel2d(image, rows, cols, minr, maxc, mode, cval);
    float bottom_left = get_pixel2d(image, rows, cols, maxr, minc, mode, cval);
    float bottom_right = get_pixel2d(image, rows, cols, maxr, maxc, mode, cval);

    top = (1 - dc) * top_left + dc * top_right;
    bottom = (1 - dc) * bottom_left + dc * bottom_right;
    out[0] = (1 - dr) * top + dr * bottom;
}

__device__ SampledCirclePoints* sample_points_from_neighborhood(unsigned int radius, unsigned int npoints) {
    float* rr;
    float* cc;

    rr = (float *) malloc(sizeof(float) * npoints);
    cc = (float *) malloc(sizeof(float) * npoints);
    
    for (int i = 0; i < npoints; ++i) {
        float point_num = (float)i / npoints;
        float circle_pos_r, circle_pos_c;
        circle_pos_r = - radius * sin(2 * M_PI * point_num);
        circle_pos_c = radius * cos(2 * M_PI * point_num);
        rr[i] = circle_pos_r;
        cc[i] = circle_pos_c;
    }

    SampledCirclePoints* sampled_points;
    sampled_points = (SampledCirclePoints*) malloc(sizeof(SampledCirclePoints));
    sampled_points->cc = cc;
    sampled_points->rr = rr;

    return sampled_points;
}

__device__ void sampled_points_delete(SampledCirclePoints* sampled_points) {
    if (sampled_points != NULL) {
        if (sampled_points->cc != NULL)
            free(sampled_points->cc);
        if (sampled_points->rr != NULL)
            free(sampled_points->rr);
        free(sampled_points);
    }
}

__global__ void lbpKernel(const uint8_t* img_data, uint32_t* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval) {
    int center_col = blockIdx.x * blockDim.x + threadIdx.x;
    int center_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (center_col >= width || center_row >= height) return;

    const int idx = center_row * width + center_col;
    float center_val = (float) get_pixel2d((uint8_t*) img_data, height, width, center_row, center_col, mode, cval);

    int prev_bit = -1;
    int first_bit = -1;
    unsigned int changes = 0;
    unsigned int sum_bits = 0;

    for (int i = 0; i < npoints; ++i){ 
        float point_num = (float)i / (float)npoints;
        float circle_pos_r = - (float)radius * sinf(2.0f * (float)M_PI * point_num);
        float circle_pos_c = (float)radius * cosf(2.0f * (float)M_PI * point_num);

        float sample_r = (float)center_row + circle_pos_r;
        float sample_c = (float)center_col + circle_pos_c;

        float sampled_value = 0.0f;

        bilinear_interpolation((uint8_t*)img_data, height, width, sample_r, sample_c, mode, (float)cval, &sampled_value);

        // printf("DEBUG %d %d %f %f %f \n", center_row, center_col, sample_r,  sample_c, sampled_value);

        int bit = (sampled_value - center_val >= 0.0f) ? 1 : 0;

        // printf("bit %d (%d, %d) \n", bit, center_row, center_col);
        
        if (i == 0) {
            first_bit = bit;
        } else {
            if (bit != prev_bit) ++changes;
        }

        // printf("prev_bit %d (%d, %d) \n", prev_bit, center_row, center_col);

        prev_bit = bit;

        // printf("prev_bit %d (%d, %d) \n", prev_bit, center_row, center_col);

        sum_bits += (unsigned int) bit;

        // printf("prev_bit %d (%d, %d) \n", sum_bits, center_row, center_col);
    }

    if (npoints > 1 && prev_bit != first_bit) ++changes;


    // printf("changes %d (%d, %d), \n", changes, center_row, center_col);

    uint32_t lbp = 0;
    if (changes <= 2) {
        lbp = sum_bits;
    } else {
        lbp = (uint32_t)(npoints + 1);
    }

    // printf("lbp %d (%d, %d) %d \n", lbp, center_row, center_col, idx);

    // printf("\n\n");

    out_feature_map[idx] = lbp;
}

// 
unsigned int* allocate_uint_img_channel_device(int n, int m) {
    size_t mat_size = n * m * sizeof(unsigned int);
    unsigned int *dev_mat;
    CUDA_CHECK(cudaMalloc(&dev_mat, mat_size));
    return dev_mat;
}

extern "C" void process_channel_with_lbp(uint8_t* img_data, uint32_t* out_feature_map, 
    int width, int height, int radius, int npoints, char mode, int cval) {

        // cudaDeviceProp deviceProp;
        // cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
        // std::cout << "Num SM: " << deviceProp.multiProcessorCount << std::endl;
        // std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;

        // Memory allocation

        uint8_t *inDevice = nullptr;
        uint32_t *outDevice = nullptr;

        size_t inBytes = (size_t)width * (size_t)height * sizeof(uint8_t);
        size_t outBytes = (size_t)width * (size_t)height * sizeof(uint32_t);


        CUDA_CHECK(cudaMalloc((void **) &inDevice, inBytes));
        CUDA_CHECK(cudaMalloc((void **) &outDevice, outBytes));

        cudaMemcpy(inDevice, img_data, inBytes, cudaMemcpyHostToDevice);

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);


        lbpKernel<<< dimGrid, dimBlock >>>(inDevice, outDevice, width, height, radius, npoints, mode, cval);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy output back to host
        CUDA_CHECK(cudaMemcpy(out_feature_map, outDevice, outBytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(inDevice));
        CUDA_CHECK(cudaFree(outDevice));
}
