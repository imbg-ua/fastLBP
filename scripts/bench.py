import fastlbp_imbg as fastlbp
import numpy as np

import memory_profiler as mprof
from time import perf_counter

def main():
    h,w = 20000, 20000
    image = fastlbp.utils.load_sample_image(h, w, 3, 'tiff', 'tmp', create=True)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//4:(h-h//4), w//4:(w-w//4)] = 1

    radii_list = fastlbp.get_radii(5)
    npoints_list = [ fastlbp.get_p_for_r(r) for r in radii_list ] 
    print(npoints_list)

    patchsize = 100

    output_abs_path = fastlbp.run_fastlbp(
        image, radii_list, npoints_list, patchsize,
        img_mask=mask,
        ncpus=10, 
        outfile_name="lbp_features.npy",  # output file name, will be in the ./data/out
        img_name="benchmark",    # human-friendly name, optional
        save_intermediate_results=False,  # do not use cache
        overwrite_output=True     # no error if output file already exists
    )

if __name__ == "__main__":
    t0 = perf_counter()
    max_mem_usage = mprof.memory_usage(main, interval=0.2, include_children=True, max_usage=True)
    t1 = perf_counter()
    print()
    print(f"MAX MEM USAGE: {max_mem_usage//(1000*1000):.3f} MB")
    print(f"EXECUTION TIME: {t1-t0:.5g} s")

