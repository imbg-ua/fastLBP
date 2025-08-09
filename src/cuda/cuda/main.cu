#include <cstdio>
#include <stdint.h>
// #include "helpers.h"
#include "common.h"

const unsigned int THREADS_PER_BLOCK = 1;
#define TILE_WIDTH 32

// static void CudaCheck(cudaError_t error, const char *file, int line) {
//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Error: %s:%d, ", file, line);
//         fprintf(stderr, "code: %d, reason: %s\n", error,
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

__global__ void lbpKernel(uint8_t* img_data, uint32_t* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval) {
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
            bilinear_interpolation(img_data, (size_t) height, (size_t) width, center_row + rr[i], center_col + cc[i], 
                mode, cval, &texture[i]);

        // threshold interpolated sampled values with the central pixel intensity
        for (size_t i = 0; i < npoints; ++i)
            if (texture[i] - img_data[center_row * width + center_col] >= 0)
                signed_texture[i] = 1;
            else
                signed_texture[i] = 0;

        // get uniform LBP code
        uint32_t lbp = 0;
        
        uint32_t changes = 0;
        for (size_t i = 0; i < npoints - 1; ++i)
            changes += (uint32_t) ((signed_texture[i] - signed_texture[i + 1]) != 0);

        if (changes <= 2)
            for (size_t i = 0; i < npoints; ++i)
                lbp += signed_texture[i];
        else
            lbp = npoints + 1;

        out_feature_map[center_row * width + center_col] = lbp;

        sampled_points_delete(sampled_points);
    }

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

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
        std::cout << "Num SM: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;

        // Memory allocation

        uint8_t *inDevice;
        uint32_t *outDevice;


        CUDA_CHECK(cudaMalloc((void **) &inDevice, width * height * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void **) &outDevice, width * height * sizeof(uint32_t)));

        cudaMemcpy(inDevice, img_data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil((float) (width + TILE_WIDTH - 1) / TILE_WIDTH), ceil((float) (height + TILE_WIDTH - 1) / TILE_WIDTH));
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

        lbpKernel<<< dimGrid, dimBlock >>>(inDevice, outDevice, width, height, radius, npoints, mode, cval);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy output back to host
        CUDA_CHECK(cudaMemcpy(out_feature_map, outDevice, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}