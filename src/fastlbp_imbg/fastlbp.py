import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union, Literal, Iterable
import os
import psutil
from collections import namedtuple
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

def get_radii(n: int=15) -> list[float]:
    """
    Get a standard sequence of radii.

    The formula is `round(1.499*1.327**(float(x)))`.
    It was coined by Ben in his initial lbp pipeline.
    """
    radius_list = [round(1.499*1.327**(float(x))) for x in range(0, n)]
    return radius_list

def get_p_for_r(r: Union[float, int, Iterable[Union[float, int]]]) -> NDArray:
    """
    Get a standard value of npoints for a single radius or a list of npoints for a list of radii.

    The formula is `np.ceil(2*np.pi*r).astype('int')`
    """
    if isinstance(r, str):
        raise TypeError()
    if isinstance(r, Iterable):
        r = np.asarray(r)
    elif isinstance(r, float) or isinstance(r, int):
        r = np.array([r])
    else: 
        raise TypeError()
    return np.ceil(2*np.pi*r).astype('int')


#####
# PIPELINE METHODS

FastlbpResult = namedtuple('FastlbpResult', 'output_abspath, patch_mask')

def run_fastlbp(img_data: ArrayLike, radii_list: ArrayLike, npoints_list: ArrayLike, 
                patchsize: int, ncpus: int, 
                img_mask=None, mask_method='any',
                max_ram=None, img_name='img', 
                outfile_name='lbp_features.npy', save_intermediate_results=True, 
                overwrite_output=False) -> FastlbpResult:
    """
    Run multiradii multichannel FastLBP feature extraction.

    - Input is a single image of shape (h,w) or (h,w,c), an integer patchsize, 
    a list of N radii and a list of N npoints.
    - Output is an np.array of shape (h//patchsize, w//patchsize, nfeatures); 
    a concatenation of N lbp outputs, one for each single lbp run, where first two axes correspond
    to the patch index (i.e. number of row and column)

    An Nth lbp run consists of computing lbp codes (with parameters R_N and P_N) for each image pixel; 
    and then computing a histogram of lbp codes for each (patchsize, patchsize) patch of the image.

    `run_fastlbp` tries to cache and use cached lbp results. 
    Cache id is img_data.shape, img_name, and patchsize.

    Computation starts with the heaviest jobs (largest radii) and ends with quickest ones.

    ## Parameters:
    
    `img_data`: np.array of shape (h,w) or (h,w,nchannels) and dtype=np.uint8

    `radii_list`: list or array of floats [R_i]

    `npoints_list`: list or array of ints [P_i]

    `patchsize`: int. A size of a square patch in pixels. This basically determines the resolution of lbp.  
    The patch size should be larger than a typical size of texture elements, but small enough to capture
    transitions between different textures.  

    `ncpus`: int. A number of parallel processes to create. More processes means less execution time
    but more memory usage. It is not recomended to set `ncpus` greater than the number of physical processors
    in your system. 
    Use value `-1` to use all available physical CPUs (determined using `psutil`).

    `img_mask`: optional, np.array of size (h,w) and dtype=np.uint8. If `img_mask` is provided, 
    the function will downscale it to a patch mask using specified `mask_method`
    and then compute for included patches only.

    `mask_method`: optional, default is 'any'. Use 'any' to include a patch if at least one its pixel is nonzero in `img_mask`.
    Use 'all' to include a patch only if all its pixels are nonzero in `img_mask`. See `utils.patchify_image_mask()` for details.

    `max_ram`: ignored, not implemented. Will be implemented in the next version.

    `img_name`: default "img". Human-friendly name to use in cache and to show in logs

    `outfile_name`: default 'lbp_features.npy'. Name of an output file. The final path is `'./data/out/{outfile_name}.npy'`.
    You cannot change the path yet, I am sorry :(
    
    `save_intermediate_results`: default True. Wether to use cache. This is espetially useful if computation
    got interrupted; cache allows to continue the process from the latest successful job. 
    Another usecase is when you need more radii and want to compute only the new ones.

    `overwrite_output`: default False. Abort fastlbp if False and output file already exists.

    ## Returns:

    `FastlbpResult(output_abspath, patch_mask)`

    `output_abspath`: str, an absolute path to the .npy file containing np.array of shape 
    (h//patchsize, w//patchsize, nfeatures) with dtype=np.uint16

    `patch_mask`: np.array of shape (h//patchsize, w//patchsize) and dtype=np.uint8.
    Contains 1 for computed patches and 0 for excluded. 
    This will equal to `utils.patchify_image_mask(img_mask, patchsize, edit_img_mask=False, method=mask_method)`

    """
    
    import time
    import pandas as pd
    from pandas import DataFrame
    from multiprocessing import Pool, shared_memory
    from .common import _features_dtype
    from .workers import __worker_fastlbp
    from .utils import patchify_image_mask

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) in [2,3]
    assert img_data.dtype == np.uint8
    
    if len(img_data.shape) == 2:
        img_data = img_data[:,:,None]

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

    assert ncpus >= -1
    max_ncpus = psutil.cpu_count(logical=False)
    if ncpus > max_ncpus:
        log.warning(f"ncpus ({ncpus}) greater than number of physical cpus ({max_ncpus})! Beware the performance issues.")
    if ncpus == -1: 
        log.info(f"ncpus == -1 so using all available physical cpus. That is, {max_ncpus} processes")
        ncpus = max_ncpus
        
    if max_ram is not None:
        log.warning("max_ram parameter is ignored!")

    # check if output file is writable

    output_fpath = os.path.join(__get_output_dir(), outfile_name)
    output_abspath = os.path.abspath(output_fpath)
    try:
        if os.path.exists(output_fpath) and not overwrite_output:
            log.error(f'run_fastlbp({pipeline_hash}): overwrite_output is False and output file {output_abspath} already exists. Aborting.')
            return FastlbpResult(output_abspath, None)
        os.makedirs(__get_output_dir(), exist_ok=True)
        if not os.access(__get_output_dir(), os.W_OK):
            log.error(f'run_fastlbp({pipeline_hash}): output dir {os.path.dirname(output_abspath)} is not writable. Aborting.')
            return FastlbpResult(output_abspath, None)
    except:
        log.error(f'run_fastlbp({pipeline_hash}): error accessing output dir {os.path.dirname(output_abspath)}. Aborting.')
        return FastlbpResult(output_abspath, None)

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
                     'output_offset', 'tmp_fpath', 
                     #'img_mask_shm_name',
                     'patch_mask_shm_name',
                    ]
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

    # copy mask to shared memory if provided.

    """
    We won't compute a patch if it has at least one pixel is ignored in img_mask.
    EVERY feature for this patch will be zero.
    Thus it is sensible to store a patch-wise mask, not the whole image mask. 
    This behavior might change in the future.

    e.g. 'exclude whole patch if at least 1 zero' vs 'include whole patch if at least 1 non-zero'
    """

    # img_mask_shm = None   # per pixel mask
    patch_mask_shm = None # per patch mask
    patch_mask_shape = (nprows, npcols)
    patch_mask = None
    if img_mask is not None:
        log.info(f"run_fastlbp({pipeline_hash}): using image mask.")
        patch_mask = patchify_image_mask(img_mask, patchsize, edit_img_mask=False, method=mask_method)
        assert patch_mask.shape == patch_mask_shape

        # img_mask_shm = shared_memory.SharedMemory(create=True, size=img_mask.nbytes)
        # img_mask_np = np.ndarray(img_mask.shape, dtype=img_mask.dtype, buffer=img_mask_shm.buf)
        # np.copyto(img_mask_np, img_mask, casting='no')
    
        patch_mask_shm = shared_memory.SharedMemory(create=True, size=patch_mask.nbytes)
        patch_mask_np = np.ndarray(patch_mask_shape, dtype=np.uint8, buffer=patch_mask_shm.buf)
        np.copyto(patch_mask_np, patch_mask, casting='no')

        log.info(f"run_fastlbp({pipeline_hash}): mask processed.")

    # create and initialize shared memory for output
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype(_features_dtype).itemsize))
    patch_features = np.ndarray(patch_features_shape, _features_dtype, buffer=patch_features_shm.buf)
    patch_features.fill(0)
    log.info(f"run_fastlbp({pipeline_hash}): shared memory created")

    jobs['img_shm_name'] = input_img_shm.name
    # jobs['img_mask_shm_name'] = img_mask_shm.name if img_mask_shm is not None else ""
    jobs['patch_mask_shm_name'] = patch_mask_shm.name if patch_mask_shm is not None else ""
    jobs['img_pixel_dtype'] = input_img_np.dtype # note: always uint8
    jobs['img_shape_0'] = input_img_np.shape[0] # nchannels
    jobs['img_shape_1'] = input_img_np.shape[1] # h
    jobs['img_shape_2'] = input_img_np.shape[2] # w
    jobs['output_shm_name'] = patch_features_shm.name

    # Log jobs before sorting
    jobs.to_csv(__get_output_dir() + f"/jobs_{img_name}.csv")
    
    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    log.info(f'run_fastlbp({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s')
    log.info(f"run_fastlbp({pipeline_hash}): jobs:")
    log.info(jobs)

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
    patch_features_shm.unlink()
    # if img_mask_shm is not None:
    #     img_mask_shm.unlink()
    if patch_mask_shm is not None:
        patch_mask_shm.unlink()

    log.info(f"run_fastlbp({pipeline_hash}): shared memory unlinked. Goodbye")
    
    return FastlbpResult(output_abspath, patch_mask)
    
def run_chunked_fastlbp():
    # TODO
    pass


    
