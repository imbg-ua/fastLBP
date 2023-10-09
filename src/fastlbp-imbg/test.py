import fastlbp
import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

def main():
    print("hewlo")

    img = Image.open("data/img1.jpg")
    img_data = np.asarray(img)
    print(img_data.shape)

    # img_data = img_data[:,:,None]
    print(img_data.shape)

    radii_list = [1,2,3]
    npoints_list = [8,12,16]
    patchsize = 100

    fastlbp.run_skimage(img_data, radii_list, npoints_list, patchsize, ncpus=4, max_ram=100, outfile_name="run_skimage_out.npy")

    results = np.load("run_skimage_out.npy", mmap_mode='r')
    print("Shape: ", results.shape)



if __name__ == '__main__':
    main()
