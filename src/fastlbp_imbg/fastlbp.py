import numpy as np
import os
import logging
logging.basicConfig()
log = logging.getLogger('fastlbp_imbg')
log.setLevel('DEBUG')

#####
# MISC ROUTINES FOR INTERNAL USAGE

def __create_pipeline_hash(method_name, *pipeline_params):
    import hashlib
    from . import __version__
    s = __version__ + ";" + method_name + ";" + (";".join([str(p) for p in pipeline_params]))
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

def run_fastlbp(img_data, radii_list, npoints_list, patchsize, ncpus, 
                img_mask=None, max_ram=None, img_name='img', 
                outfile_name='lbp_features.npy', save_intermediate_results=True, overwrite_output=False):    
    import time
    import pandas as pd
    from pandas import DataFrame
    from multiprocessing import Pool, shared_memory
    from .common import _features_dtype
    from .workers import __worker_fastlbp

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) == 3
    assert img_data.dtype == np.uint8
    
    if img_mask is not None:
        assert img_mask.shape == img_data.shape[:2]
        assert img_mask.dtype == np.uint8

    t = time.perf_counter()
    
    log.info('run_fastlbp: initial setup...')

    img_name = __sanitize_img_name(img_name)
    outfile_name = __sanitize_outfile_name(outfile_name)
    # data_hash = hashlib.sha1(img_data.data).hexdigest()

    # this way pipelines with different ncpus/radii/npoints can reuse tmp files if patchsize, img name and version are the same 
    pipeline_hash = __create_pipeline_hash("fastlbp", [str(img_data.shape), patchsize, img_name])
    pipeline_name = f"{img_name}-fastlbp-{pipeline_hash}"

    log.info('run_fastlbp: params:')
    log.info("img_shape, radii_list, npoints_list, patchsize, ncpus, max_ram, img_name")
    log.info(f"{img_data.shape}, {radii_list}, {npoints_list}, {patchsize}, {ncpus}, {max_ram}, {img_name}")
    log.info(f"outfile_name={outfile_name}, save_intermediate_results={save_intermediate_results}, overwrite_output={overwrite_output}")
    log.info(f"pipeline hash is {pipeline_hash}")

    # check if output file is writable

    output_fpath = os.path.join(__get_output_dir(), outfile_name)
    output_abspath = os.path.abspath(output_fpath)
    try:
        if os.path.exists(output_fpath) and not overwrite_output:
            log.error(f'run_fastlbp({pipeline_hash}): overwrite_output is False and output file {output_abspath} already exists. Aborting.')
            return output_abspath
        os.makedirs(__get_output_dir(), exist_ok=True)
        if not os.access(__get_output_dir(), os.W_OK):
            log.error(f'run_fastlbp({pipeline_hash}): output dir {os.path.dirname(output_abspath)} is not writable. Aborting.')
            return output_abspath
    except:
        log.error(f'run_fastlbp({pipeline_hash}): error accessing output dir {os.path.dirname(output_abspath)}. Aborting.')
        return output_abspath

    log.info(f'run_fastlbp({pipeline_hash}): initial setup took {time.perf_counter()-t:.5g}s')
    log.info(f'run_fastlbp({pipeline_hash}): creating a list of jobs...')
    t = time.perf_counter()

    # method-specific params

    h,w,nchannels = img_data.shape
    nprows, npcols = h//patchsize, w//patchsize
    nfeatures_cumsum = np.cumsum(np.array(npoints_list)+2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    # create a list of jobs
    jobs = DataFrame(
        index=pd.MultiIndex.from_product(
            [channel_list, radii_list], names=['channel', 'radius']), 
            columns=['channel','radius','img_name','label','npoints','patchsize','img_shm_name',
                     'img_pixel_dtype','img_shape_0','img_shape_1','img_shape_2', 'output_shm_name', 
                     'output_offset', 'tmp_fpath', 'img_mask_shm_name']
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

    # Prepare contigous array.
    # Channels will go first. Then h and w.
    img_data = np.ascontiguousarray(np.moveaxis(img_data, (0,1,2), (1,2,0)))
    
    log.info(f"run_fastlbp({pipeline_hash}): creating shared memory")
    # create shared memory for input image
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)

    # copy image to shared memory 
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    np.copyto(input_img_np, img_data, casting='no')

    # copy mask to shared memory if provided
    img_mask_shm = None
    if img_mask is not None:
        log.info(f"run_fastlbp({pipeline_hash}): using image mask.")
        img_mask_shm = shared_memory.SharedMemory(create=True, size=img_mask.nbytes)
        img_mask_np = np.ndarray(img_mask.shape, dtype=img_mask.dtype, buffer=img_mask_shm.buf)
        np.copyto(img_mask_np, img_mask, casting='no')

    # create and initialize shared memory for output
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype(_features_dtype).itemsize))
    patch_features = np.ndarray(patch_features_shape, _features_dtype, buffer=patch_features_shm.buf)
    patch_features.fill(0)
    log.info(f"run_fastlbp({pipeline_hash}): shared memory created")

    jobs['img_shm_name'] = input_img_shm.name
    jobs['img_mask_shm_name'] = img_mask_shm.name if img_mask_shm is not None else ""
    jobs['img_pixel_dtype'] = input_img_np.dtype # note: always uint8
    jobs['img_shape_0'] = input_img_np.shape[0] # nchannels
    jobs['img_shape_1'] = input_img_np.shape[1] # h
    jobs['img_shape_2'] = input_img_np.shape[2] # w
    jobs['output_shm_name'] = patch_features_shm.name

    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    log.info(f'run_fastlbp({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s')
    log.info(f"run_fastlbp({pipeline_hash}): jobs:")
    log.info(jobs)
    jobs.to_csv(__get_output_dir() + f"/jobs_{img_name}.csv")

    assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f'run_fastlbp({pipeline_hash}): start computation')
    t0 = time.perf_counter()
    with Pool(ncpus) as pool:
        jobs_results = pool.map(func=__worker_fastlbp, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f'run_fastlbp({pipeline_hash}): computation finished in {t_elapsed:.5g}s. Start saving')

    # save results

    np.save(output_fpath, patch_features)
    log.info(f'run_fastlbp({pipeline_hash}): saving finished to {output_fpath}')
    
    input_img_shm.unlink()
    if img_mask_shm is not None:
        img_mask_shm.unlink()
    patch_features_shm.unlink()

    log.info(f"run_fastlbp({pipeline_hash}): shared memory unlinked. Goodbye")
    
    return output_abspath
    
def run_chunked_fastlbp():
    # TODO
    pass


    
