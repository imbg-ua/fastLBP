import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

### if installed as pip package 
# import fastlbp_imbg as fastlbp

### if using fastlbp.py 
import fastlbp

import utils

def main():
    print("hewlo")

    img = utils.load_sample_image(12000, 12000, 1)
    img_data = np.asarray(img)
    print(img_data.shape)

    h, w = img_data.shape[0], img_data.shape[1]
    img_mask = np.zeros((h,w), dtype=np.uint8)
    # img_mask[h//4:3*h//4, w//4:3*w//4] = 1
    img_mask[:] = 1

    # to add 3rd dimension
    if len(img_data.shape) == 2:
        img_data = img_data[:,:,None]
        print(img_data.shape)

    radii_list = [1,2,3,4,5]
    npoints_list = [ fastlbp.get_p_for_r(r) for r in radii_list ] 
    print(npoints_list)

    patchsize = 16

    output_abs_path = fastlbp.run_skimage(
        img_data, radii_list, npoints_list, patchsize, 
        ncpus=4, 
        outfile_name="masked_lbp.npy",  # output file name, will be in the ./data/out
        img_name="masked",    # human-friendly name, optional
        max_ram=100,        # unused yet
        save_intermediate_results=False,    # perform computations in RAM only
        overwrite_output=True    # error if output file already exists
    )

    # Uncomment this for basic output viz

    # results = np.load(output_abs_path, mmap_mode='r')
    # print(f"Shape of {output_abs_path}: ", results.shape)

    # import matplotlib.pyplot as plt
    # N_samples = 30
    # np.random.seed(time.time_ns() & (2**32 - 1))

    # rand_x = np.random.randint(0, results.shape[0], N_samples)
    # rand_y = np.random.randint(0, results.shape[1], N_samples)
    # results_subset = np.zeros((N_samples,results.shape[2]))
    # y_ticks_labels = []
    # for i,(x,y) in enumerate(zip(rand_x,rand_y)):
    #     results_subset[i,:] = (results[x,y,:])
    #     y_ticks_labels.append(f"({x},{y})")

    # features_label = []
    # for p in npoints_list:
    #     features_label += list(range(0,p+2))
    # features_label = np.array(features_label)

    # plt.imshow(results_subset)
    # plt.yticks(np.arange(N_samples), y_ticks_labels)
    # plt.ylabel("patches")
    # xticks = np.arange(0, results.shape[2]-1)
    # plt.xticks(xticks)
    # plt.xlabel("features")
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
