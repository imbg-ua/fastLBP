#ifdef __cplusplus
extern "C" {
#endif

void process_channel_with_lbp(const unsigned char* img_data, unsigned int* out_feature_map, 
    int width, int height, int radius, int npoints, char mode, int cval);

#ifdef __cplusplus
}
#endif
