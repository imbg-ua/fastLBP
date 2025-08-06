#include "helpers.h"
#define _USE_MATH_DEFINES
#include <math.h>


int get_pixel2d(int* img, size_t rows, size_t cols, size_t r, size_t c, char mode, float cval) {

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

inline void bilinear_interpolation(int* image, size_t rows, size_t cols, float r, float c, char mode, float cval, float* out) {
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

SampledCirclePoints* sample_points_from_neighborhood(unsigned int radius, unsigned int npoints) {
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

void sampled_points_delete(SampledCirclePoints* sampled_points) {
    if (sampled_points != NULL) {
        if (sampled_points->cc != NULL)
            free(sampled_points->cc);
        if (sampled_points->rr != NULL)
            free(sampled_points->rr);
        free(sampled_points);
    }
}