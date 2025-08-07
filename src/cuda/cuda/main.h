#ifndef __MAIN_PROCESS_CHANNEL_WITH_LBP__
#define __MAIN_PROCESS_CHANNEL_WITH_LBP__

void* process_channel_with_lbp(unsigned int* img_data, unsigned int* out_feature_map, int width, int height, int radius, int npoints, char mode, int cval);

#endif