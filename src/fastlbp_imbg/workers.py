import logging
log = logging.getLogger('fastlbp_imbg')
log.setLevel('DEBUG')

import time
import os
import numpy as np
from multiprocessing import shared_memory

from .common import _features_dtype
from .utils import get_patch, get_reduced_hist_masks, reduce_histogram
from .lbp import (
    uniform_lbp_uint8, 
    uniform_lbp_uint8_masked, 
    uniform_lbp_uint8_patch_masked,
)

def __worker_fastlbp(args):
    row_id, job = args
    tmp_fpath = job['tmp_fpath']

    pid = os.getpid()
    jobname = job['label']
    log.info(f"run_fastlbp: worker {pid}: starting job {jobname}")

    try:
        t0 = time.perf_counter()

        shape = job['img_shape_0'], job['img_shape_1'], job['img_shape_2']
        total_nfeatures = job['total_nfeatures']
        output_offset = job['output_offset']
        
        patchsize = job['patchsize']
        nchannels,h,w = shape
        nprows, npcols = h//patchsize, w//patchsize
        
        # !!! magic const
        if job['npoints'] < 15:
            job_nfeatures = job['npoints']+2
        else:
            job_nfeatures = 4
        job_patch_histograms_shape = (nprows, npcols, job_nfeatures)

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job['output_shm_name'])
        all_patch_histograms = np.ndarray(
            (nprows, npcols, total_nfeatures), dtype=_features_dtype, buffer=output_shm.buf)
        job_patch_histograms = all_patch_histograms[:,:,output_offset:(output_offset+job_nfeatures)]
        
        # Try to use cached data
        cached_result_mm = None
        if not tmp_fpath:
            log.debug(f"run_fastlbp: worker {jobname}({pid}): skipping cache")
        else:
            try:
                cached_result_mm = np.memmap(tmp_fpath, dtype=_features_dtype, mode='r', shape=job_patch_histograms_shape)
            except:
                cached_result_mm = None
                log.debug(f"run_fastlbp: worker {jobname}({pid}): no usable cache")
        
        if cached_result_mm is not None:
            # Use cache and return
            
            log.info(f"run_fastlbp: worker {jobname}({pid}): cache found! copying to output.")
            job_patch_histograms = cached_result_mm

        else: 
             # Compute LBP

            img_data_shm = shared_memory.SharedMemory(name=job['img_shm_name'])
            img_data = np.ndarray(shape, dtype=job['img_pixel_dtype'], buffer=img_data_shm.buf)
            
            img_channel = img_data[job['channel']]
            assert img_channel.flags.c_contiguous
            assert img_channel.dtype == np.uint8
            
            using_image_mask = 'img_mask_shm_name' in job and job['img_mask_shm_name']
            using_patch_mask = 'patch_mask_shm_name' in job and job['patch_mask_shm_name']

            if using_image_mask:
                log.debug(f"run_fastlbp: worker {jobname}({pid}): using image mask")
                img_mask_shm = shared_memory.SharedMemory(name=job['img_mask_shm_name'])
                img_mask = np.ndarray((h,w), dtype=np.uint8, buffer=img_mask_shm.buf)
                lbp_results = uniform_lbp_uint8_masked(
                    image=img_channel, mask=img_mask, 
                    P=job['npoints'], R=job['radius']
                )
                img_mask_shm.close()
            elif using_patch_mask:
                log.debug(f"run_fastlbp: worker {jobname}({pid}): using patch mask")
                patch_mask_shm = shared_memory.SharedMemory(name=job['patch_mask_shm_name'])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)
                lbp_results = uniform_lbp_uint8_patch_masked(
                    image=img_channel, patch_mask=patch_mask, patchsize=patchsize, 
                    P=job['npoints'], R=job['radius']
                )
            else:
                # if no mask is provided
                log.debug(f"run_fastlbp: worker {jobname}({pid}): do not use mask")
                lbp_results = uniform_lbp_uint8(image=img_channel, P=job['npoints'], R=job['radius'])
            
            assert lbp_results.dtype == _features_dtype

            img_data_shm.close()

            # flat, corner, edge, nonuniform
            reduced_hist_masks = get_reduced_hist_masks(job['npoints'])

            for pr in range(nprows):
                for pc in range(npcols):
                    if using_patch_mask and patch_mask[pr,pc] == 0:
                        job_patch_histograms[pr,pc,:] = 0
                    else:
                        hist = np.bincount(
                            get_patch(lbp_results, patchsize, pr, pc).flat, 
                            minlength=job_nfeatures
                            )
                        
                        # UNTESTED
                        # reduced feature vector. unoptimized, slow, and a mess but should work
                        reduced_hist = reduce_histogram(hist, *reduced_hist_masks)
                        # \UNTESTED

                        job_patch_histograms[pr,pc,:] = reduced_hist

            if using_patch_mask:
                patch_mask_shm.close()
            if tmp_fpath:
                try:
                    os.makedirs( os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_patch_histograms)
                except:
                    log.warning(f"run_fastlbp: worker {jobname}({pid}): computation successful, but cannot save tmp file")

        output_shm.close()

        log.info(f"run_fastlbp: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")
    except Exception as e:
        log.error(f"run_fastlbp: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    return 0
