import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

### if installed as pip package 
# import fastlbp_imbg as fastlbp

### if using fastlbp.py 
import fastlbp


# TODO: run_skimage has changed its signature!!!
# TODO: run_skimage now reads img_data on its own

def main():
    print("hewlo")

    img = Image.open("data/bark.tiff")
    img_data = np.asarray(img)
    print(img_data.shape)

    # to add 3rd dimension
    img_data = img_data[:,:,None]
    print(img_data.shape)

    radii_list = [1,2,3,4]
    npoints_list = [ fastlbp.get_p_for_r(r) for r in radii_list ] 
    print(npoints_list)

    patchsize = 16

    output_abs_path = fastlbp.run_skimage(
        img_data, radii_list, npoints_list, patchsize, 
        ncpus=4, 
        outfile_name="bark_lbp.npy",  # output file name, will be in the ./data/out
        img_name="bark",    # human-friendly name, optional
        max_ram=100,        # unused yet
        save_intermediate_results=False,    # perform computations in RAM only
        overwrite_output=False    # error if output file already exists
    )

    results = np.load(output_abs_path, mmap_mode='r')
    print(f"Shape of {output_abs_path}: ", results.shape)

    # Uncomment this for basic output viz

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
