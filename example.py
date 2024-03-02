import numpy as np
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

### if installed as pip package 
import fastlbp_imbg as fastlbp

def main():
    print(f"hewlo. running fastlbp ver. {fastlbp.__version__}")

    # Will a create random input image in ./tmp if not exists yet
    img_data = fastlbp.load_sample_image(5000,5000,3,'tiff',create=True)
    print(img_data.shape)

    # Alternatively, load an image from existing file
    # img = Image.open('data/bark.tiff') 
    # img_data = np.asarray(img)

    # if len(img_data.shape) == 2: 
    #     img_data = img_data[:,:,None]
    #     print(img_data.shape)

    radii_list = [1,2,3,4,5]
    npoints_list = fastlbp.get_p_for_r(radii_list) 
    print(npoints_list)

    patchsize = 16

    features_details = fastlbp.get_all_features_details(3, radii_list, npoints_list)
    print("\n".join(map(str,features_details)))

    output_abs_path, mask = fastlbp.run_fastlbp(
        img_data, radii_list, npoints_list, patchsize, 
        ncpus=-1, 
        outfile_name="lbp_features.npy",  # output file name, will be in the ./data/out
        img_name="whitenoise",    # human-friendly name, optional
        save_intermediate_results=False,  # do not use cache
        overwrite_output=True     # no error if output file already exists
    )

    print("output_abs_path", output_abs_path)
    print("mask", mask)

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
