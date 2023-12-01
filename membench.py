import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

### if installed as pip package 
import fastlbp_imbg as fastlbp

### if using fastlbp.py 
# import fastlbp

def main():
    print("hewlo")

    img_data = fastlbp.utils.load_sample_image(10000,10000,1,'tiff')
    print(img_data.shape)

    radii_list = [5]
    npoints_list = [ fastlbp.get_p_for_r(r) for r in radii_list ] 
    print(npoints_list)

    patchsize = 5000

    output_abs_path = fastlbp.run_skimage(
        img_data, radii_list, npoints_list, patchsize, 
        ncpus=4, 
        outfile_name="membench.npy",  # output file name, will be in the ./data/out
        img_name="membench",    # human-friendly name, optional
        max_ram=100,        # unused yet
        save_intermediate_results=False,    # perform computations in RAM only
        overwrite_output=True 
    )



if __name__ == '__main__':
    main()
