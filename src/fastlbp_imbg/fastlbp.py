import numpy as np
import skimage as ski
from pandas import DataFrame
import pandas as pd

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import time
import os
from multiprocessing import Pool, shared_memory

import hashlib

from .lbp import uniform_local_binary_pattern

#####
# PIPELINE WORKERS FOR INTERNAL USAGE

def __worker_skimage(args):
    row_id, job = args
    tmp_fpath = job['tmp_fpath']

    pid = os.getpid()
    jobname = job['label']
    log.info(f"run_skimage: worker {pid}: starting job {jobname}")

    try:
        t0 = time.perf_counter()

        shape = job['img_shape_0'], job['img_shape_1'], job['img_shape_2']
        total_nfeatures = job['total_nfeatures']
        output_offset = job['output_offset']
        
        patchsize = job['patchsize']
        h,w,nchannels = shape
        nprows, npcols = h//patchsize, w//patchsize
        
        job_nfeatures = job['npoints']+2
        job_patch_histograms_shape = (nprows, npcols, job_nfeatures)

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job['output_shm_name'])
        all_patch_histograms = np.ndarray(
            (nprows, npcols, total_nfeatures), dtype=np.uint32, buffer=output_shm.buf)
        job_patch_histograms = all_patch_histograms[:,:,output_offset:(output_offset+job_nfeatures)]
        
        # if ((0 < job_patch_histograms) & (job_patch_histograms < np.iinfo(np.uint32).max)).any():
        #     log.warning(f"run_skimage: worker {jobname}({pid}): job_patch_histograms is not zero! Possible memory corruption")

        # log.info(f"run_skimage: worker {jobname}({pid}): job_nfeatures={job_nfeatures}, job_patch_histograms_shape={job_patch_histograms_shape}, job_interval=[{output_offset},{output_offset+job_nfeatures-1}]")

        # Try to use cached data
        cached_result_mm = None
        if not tmp_fpath:
            log.info(f"run_skimage: worker {jobname}({pid}): skipping cache")
        else:
            try:
                cached_result_mm = np.memmap(tmp_fpath, dtype=np.uint32, mode='r', shape=job_patch_histograms_shape)
            except:
                cached_result_mm = None
                log.info(f"run_skimage: worker {jobname}({pid}): no usable cache")
        
        if cached_result_mm is not None:
            # Use cache and return
            
            log.info(f"run_skimage: worker {jobname}({pid}): cache found! copying to output.")
            job_patch_histograms = cached_result_mm

        else: 
            # Compute LBP

            img_data_shm = shared_memory.SharedMemory(name=job['img_shm_name'])
            img_data = np.ndarray(shape, dtype=job['img_pixel_dtype'], buffer=img_data_shm.buf)
            img_mask_shm = shared_memory.SharedMemory(name=job['mask_shm_name'])
            img_mask = np.ndarray((h,w), dtype=np.uint8, buffer=img_mask_shm.buf)
            
            lbp_results = uniform_local_binary_pattern(
                image=img_data[:,:,job['channel']], P=job['npoints'], R=job['radius'], mask=img_mask
            ) #.astype(np.uint16)
            
            # log.info(f"run_skimage: worker {jobname}({pid}): image: min={img_data[:,:,job['channel']].min()} max={img_data[:,:,job['channel']].max()} avg={img_data[:,:,job['channel']].mean()}")
            # log.info(f"run_skimage: worker {jobname}({pid}): lbp codes: min={lbp_results.min()} max={lbp_results.max()}")

            img_data_shm.close()
            img_mask_shm.close()

            for i in range(nprows):
                for j in range(npcols):
                    hist = np.bincount(
                        lbp_results[(i*patchsize):((i+1)*patchsize), (j*patchsize):((j+1)*patchsize)].flat, 
                        minlength=job_nfeatures
                        )
                    job_patch_histograms[i,j,:] = hist

            if tmp_fpath:
                try:
                    os.makedirs( os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_patch_histograms)
                except:
                    log.warning(f"run_skimage: worker {jobname}({pid}): computation successful, but cannot save tmp file")

        

        # log.info(f"run_skimage: worker {jobname}({pid}): feature vector for (4,3) = {job_patch_histograms[4,3,:]}")
        
        output_shm.close()

        log.info(f"run_skimage: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")
    except Exception as e:
        log.error(f"run_skimage: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    return 0


#####
# MISC ROUTINES FOR INTERNAL USAGE

def __create_pipeline_hash(method_name, *pipeline_params):
    s = method_name + ";" + (";".join([str(p) for p in pipeline_params]))
    return hashlib.sha1(s.encode('utf-8'), usedforsecurity=False).hexdigest()[:7]
def __sanitize_img_name(img_name):
    return img_name.replace('.','_').replace('-','_')
def __sanitize_outfile_name(outfile_name):
    if outfile_name.endswith(".npy"): return outfile_name
    return outfile_name + ".npy"
def __get_output_dir():
    return os.path.join("data/out")
def __get_tmp_dir(pipeline_name):
    return os.path.join("data/tmp/", pipeline_name)

#####
# PUBLIC UTILS

# get radii
# src: Ben's original pipeline
def get_radii(n=15):
    radius_list = [round(1.499*1.327**(float(x))) for x in range(0, n)]
    return radius_list

# get number of LBP sectors for a given radius
# src: Ben's original pipeline
def get_p_for_r(r):
    return np.ceil(2*np.pi*r).astype('int')


#####
# PIPELINE METHODS

def run_skimage(img_data, radii_list, npoints_list, patchsize, ncpus, 
                img_mask=None,
                max_ram=None, img_name='img', 
                outfile_name='lbp_features_skimage.npy', save_intermediate_results=True, overwrite_output=False):
    
    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) == 3
    
    t = time.perf_counter()
    
    log.info('run_skimage: initial setup...')

    img_name = __sanitize_img_name(img_name)
    outfile_name = __sanitize_outfile_name(outfile_name)
    # data_hash = hashlib.sha1(img_data.data).hexdigest()

    pipeline_params = [str(img_data.shape), radii_list, npoints_list, patchsize, ncpus, max_ram, img_name]
    pipeline_hash = __create_pipeline_hash("skimage", pipeline_params)
    pipeline_name = f"{img_name}-skimage-{pipeline_hash}"

    log.info('run_skimage: params:')
    log.info('[data_hash, radii_list, npoints_list, patchsize, ncpus, max_ram, img_name]')
    log.info(pipeline_params)

    # check if output file is writable

    output_fpath = os.path.join(__get_output_dir(), outfile_name)
    output_abspath = os.path.abspath(output_fpath)
    try:
        if os.path.exists(output_fpath) and not overwrite_output:
            log.error(f'run_skimage({pipeline_hash}): overwrite_output is False and output file {output_abspath} already exists. Aborting.')
            return output_abspath
        os.makedirs(__get_output_dir(), exist_ok=True)
        if not os.access(__get_output_dir(), os.W_OK):
            log.error(f'run_skimage({pipeline_hash}): output dir {os.path.dirname(output_abspath)} is not writable. Aborting.')
            return output_abspath
    except:
        log.error(f'run_skimage({pipeline_hash}): error accessing output dir {os.path.dirname(output_abspath)}. Aborting.')
        return output_abspath

    log.info(f'run_skimage({pipeline_hash}): initial setup took {time.perf_counter()-t:.5g}s')
    log.info(f'run_skimage({pipeline_hash}): creating a list of jobs...')
    t = time.perf_counter()

    # method-specific params

    h,w,nchannels = img_data.shape
    nprows, npcols = h//patchsize, w//patchsize
    nfeatures_cumsum = np.cumsum(np.array(npoints_list)+2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    if img_mask is not None:
        assert len(img_mask.shape) == 2, "image mask has invalid dimensions (2 required)"
        assert img_mask.shape[0] == h and img_mask.shape[1] == w, "image mask shape is incompatible with image"
        assert img_mask.dtype == np.uint8, "only uint8 masks are supported"
    else:
        img_mask = np.zeros((h,w), dtype=np.uint8)


    # create a list of jobs
    jobs = DataFrame(
        index=pd.MultiIndex.from_product(
            [channel_list, radii_list], names=['channel', 'radius']), 
            columns=['channel','radius','img_name','label','npoints','patchsize','img_shm_name',
                     'img_pixel_dtype','img_shape_0','img_shape_1','img_shape_2', 
                     'mask_shm_name', 'output_shm_name', 'output_offset', 'tmp_fpath']
        )
    jobs['img_name'] = img_name

    channel_output_offset = 0
    for c in channel_list: 
        jobs.loc[c,'channel'] = c
        jobs.loc[c,'radius'] = radii_list
        jobs.loc[c,'npoints'] = npoints_list
        jobs.loc[c,'output_offset'] = channel_output_offset + np.hstack([[0],nfeatures_cumsum[:-1]])
        channel_output_offset += nfeatures_per_channel
    
    jobs['label'] = jobs.apply(
        lambda row: f"{img_name}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}", axis='columns')
    jobs['patchsize'] = patchsize

    if save_intermediate_results:
        base_tmp_path = __get_tmp_dir(pipeline_name)
        jobs['tmp_fpath'] = jobs.apply(
            lambda row: f"{base_tmp_path}/{row['label']}.npy", axis='columns')
    else: 
        jobs['tmp_fpath'] = ""

    total_nfeatures = nfeatures_per_channel * len(channel_list)
    patch_features_shape = (nprows, npcols, total_nfeatures)
    jobs['total_nfeatures'] = total_nfeatures

    # create shared memory for all processes

    log.info(f"run_skimage({pipeline_hash}): creating shared memory")
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)
    input_mask_shm = shared_memory.SharedMemory(create=True, size=img_mask.nbytes)
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype('uint32').itemsize))
    
    # copy image to shared memory 
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    input_img_np[:] = img_data[:]

    # copy mask to shared memory
    input_mask_np = np.ndarray(img_mask.shape, img_mask.dtype, input_mask_shm.buf)
    input_mask_np[:] = img_mask[:]
    
    # fill output with an invalid val
    patch_features = np.ndarray(patch_features_shape, np.uint32, buffer=patch_features_shm.buf)
    patch_features.fill(np.iinfo(np.uint32).max)

    jobs['img_shm_name'] = input_img_shm.name
    jobs['img_pixel_dtype'] = img_data.dtype
    jobs['img_shape_0'] = img_data.shape[0]
    jobs['img_shape_1'] = img_data.shape[1]
    jobs['img_shape_2'] = img_data.shape[2]
    jobs['mask_shm_name'] = input_mask_shm.name
    jobs['output_shm_name'] = patch_features_shm.name

    # sort descending to compute the heaviest jobs first
    jobs.sort_index(level=1, ascending=False, inplace=True)

    log.info(f'run_skimage({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s')
    log.info(f"run_skimage({pipeline_hash}): jobs:")
    log.info(jobs)
    jobs.to_csv(__get_output_dir() + f"/jobs_{img_name}.csv")

    assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f'run_skimage({pipeline_hash}): start computation')
    t0 = time.perf_counter()
    with Pool(ncpus) as pool:
        _ = pool.map(func=__worker_skimage, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f'run_skimage({pipeline_hash}): computation finished in {t_elapsed:.5g}s. Start saving')

    # save results

    np.save(output_fpath, patch_features)
    log.info(f'run_skimage({pipeline_hash}): saving finished to {output_fpath}')
    
    input_mask_shm.unlink()
    input_img_shm.unlink()
    patch_features_shm.unlink()

    log.info(f"run_skimage({pipeline_hash}): shared memory unlinked. Goodbye")
    
    return output_abspath
    
def run_chunked_skimage():
    # TODO
    pass


    
