import numpy as np
import numpy.typing as npt
import skimage as ski
from PIL import Image
from collections import namedtuple
from pandas import DataFrame
import pandas as pd

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import time
import os
from multiprocessing import Pool, shared_memory

def __worker_skimage(args):
    row_id, job = args

    pid = os.getpid()
    jobname = job['label']
    log.info(f"run_skimage: do_job worker {pid}: starting job {jobname}")
    t0 = time.perf_counter()

    shape = job['img_shape_0'], job['img_shape_1'], job['img_shape_2']
    total_nfeatures = job['total_nfeatures']
    output_offset = job['output_offset']
    
    patchsize = job['patchsize']
    h,w,nchannels = shape
    nprows, npcols = h//patchsize, w//patchsize

    img_data_shm = shared_memory.SharedMemory(name=job['img_shm_name'])
    img_data = np.ndarray(shape, dtype=job['img_pixel_dtype'], buffer=img_data_shm.buf)
    
    output_shm = shared_memory.SharedMemory(name=job['output_shm_name'])
    all_patch_histograms = np.ndarray((nprows, npcols, total_nfeatures), dtype=np.uint32, buffer=output_shm.buf)

    job_nfeatures = job['npoints']+2
    job_patch_histograms = all_patch_histograms[:,:,output_offset:(output_offset+job_nfeatures)]

    nfeatures = job['npoints'] + 2

    lbp_results = ski.feature.local_binary_pattern(img_data[:,:,job['channel']], method='uniform', P=job['npoints'], R=job['radius']).astype(np.uint16)
    for i in range(nprows):
        for j in range(npcols):
            hist = np.bincount(lbp_results[(i*patchsize):(i*(patchsize+1)), (j*patchsize):(j*(patchsize+1))].flat, minlength=nfeatures)
            job_patch_histograms[i,j,:] = hist
    
    output_shm.close()
    img_data_shm.close()

    log.info(f"run_skimage: do_job worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.3f}s")

    return row_id


def run_skimage(img_data, radii_list, npoints_list, patchsize, ncpus, max_ram=None, img_name='img', outfile_name='lbp_skimage.npy'):
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) == 3

    h,w,nchannels = img_data.shape
    nprows, npcols = h//patchsize, w//patchsize
    npoints_cumsum = np.cumsum(npoints_list)
    nfeatures_per_channel = npoints_cumsum[-1]
    channel_list = range(nchannels)

    jobs = DataFrame(
        index=pd.MultiIndex.from_product(
            [channel_list, radii_list], names=['channel', 'radius']), 
            columns=['channel','radius','img_name','label','npoints','patchsize','img_shm_name','img_pixel_dtype','img_shape_0','img_shape_1','img_shape_2', 'output_shm_name', 'output_offset']
        )
    jobs['img_name'] = img_name

    channel_output_offset = 0
    for c in channel_list: 
        jobs.loc[c,'channel'] = c
        jobs.loc[c,'radius'] = radii_list
        jobs.loc[c,'npoints'] = npoints_list
        jobs.loc[c,'output_offset'] = channel_output_offset + np.hstack([[0],npoints_cumsum[:-1]])
        channel_output_offset += nfeatures_per_channel
    
    jobs['label'] = jobs.apply(lambda row: f"{img_name}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}", axis='columns')
    jobs['patchsize'] = patchsize

    total_nfeatures = (npoints_cumsum[-1]+2) * len(channel_list)
    patch_features_shape = (nprows, npcols, total_nfeatures)
    jobs['total_nfeatures'] = total_nfeatures

    log.info(f"run_skimage: creating shared memory")
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)
    patch_features_shm = shared_memory.SharedMemory(create=True, size=(int(np.prod(patch_features_shape)) * np.dtype('uint32').itemsize))
    
    patch_features = np.ndarray(patch_features_shape, np.uint32, buffer=patch_features_shm.buf)

    jobs['img_shm_name'] = input_img_shm.name
    jobs['img_pixel_dtype'] = img_data.dtype
    jobs['img_shape_0'] = img_data.shape[0]
    jobs['img_shape_1'] = img_data.shape[1]
    jobs['img_shape_2'] = img_data.shape[2]
    jobs['output_shm_name'] = patch_features_shm.name

    log.info(f"run_skimage: jobs:")
    log.info(jobs)

    assert jobs.isna().sum().sum() == 0

    log.info('run_skimage: start computation')
    t0 = time.perf_counter()
    with Pool(ncpus) as pool:
        jobs_results = pool.map(func=__worker_skimage, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f'run_skimage: computation finished in {t_elapsed:.3f}s. Start saving')

    np.save(outfile_name, patch_features)
    log.info(f'run_skimage: saving finished to {outfile_name}')
    
    input_img_shm.unlink()
    patch_features_shm.unlink()

    log.info("run_skimage: shared memory unlinked. Goodbye")

    
def run_chunked_skimage():
    # TODO
    pass


    
