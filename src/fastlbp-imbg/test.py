import fastlbp
import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

import matplotlib.pyplot as plt
import time

def main():
    print("hewlo")

    img = Image.open("data/bark.tiff")
    img_data = np.asarray(img)[:,:,None]
    print(img_data.shape)

    # img_data = img_data[:,:,None]
    print(img_data.shape)

    radii_list = [1,2,3,4]
    npoints_list = [8,12,16,24] # NOTE: SHITTY VALUES TO MAKE THINGS FASTER
    patchsize = 16

    output_abs_path = fastlbp.run_skimage(
        img_data, radii_list, npoints_list, patchsize, 
        ncpus=4, max_ram=100, outfile_name="bark_lbp.npy", img_name="bark.tiff"
        )

    results = np.load(output_abs_path, mmap_mode='r')
    print(f"Shape of {output_abs_path}: ", results.shape)

    N_samples = 30
    np.random.seed(time.time_ns() & (2**32 - 1))

    rand_x = np.random.randint(0, results.shape[0], N_samples)
    rand_y = np.random.randint(0, results.shape[1], N_samples)
    results_subset = np.zeros((N_samples,results.shape[2]))
    y_ticks_labels = []
    for i,(x,y) in enumerate(zip(rand_x,rand_y)):
        results_subset[i,:] = (results[x,y,:])
        y_ticks_labels.append(f"({x},{y})")

    features_label = []
    for p in npoints_list:
        features_label += list(range(0,p+2))
    features_label = np.array(features_label)

    plt.imshow(results_subset)
    plt.yticks(np.arange(N_samples), y_ticks_labels)
    plt.ylabel("patches")
    xticks = np.arange(0, results.shape[2]-1)
    plt.xticks(xticks)
    plt.xlabel("features")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
