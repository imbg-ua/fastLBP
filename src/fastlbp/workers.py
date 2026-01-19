import logging

log = logging.getLogger("fastlbp")
log.setLevel("DEBUG")

import os
import time
from multiprocessing import shared_memory

import numpy as np

# import lbp_cuda # import cuda worker from a separate package
try:
    from ._lbp_gpu import cuda_lbp  # noqa: F401

    _HAS_CUDA_EXT = True
except Exception:
    _HAS_CUDA_EXT = False

from .common import _features_dtype, _raw_features_dtype
from .lbp import (
    uniform_lbp_uint8,
    uniform_lbp_uint8_masked,
    uniform_lbp_uint8_padded_absolute,
    uniform_lbp_uint8_padded_absolute_masked,
    uniform_lbp_uint8_padded_absolute_patch_masked,
    uniform_lbp_uint8_patch_masked,
)
from .utils import get_padded_region, get_patch


def __worker_fastlbp(args):
    row_id, job = args
    tmp_fpath = job["tmp_fpath"]
    tmp_fpath_pixel = job["tmp_fpath_pixel"]

    pid = os.getpid()
    jobname = job["label"]
    log.info(f"run_fastlbp: worker {pid}: starting job {jobname}")

    try:
        t0 = time.perf_counter()

        shape = job["img_shape_0"], job["img_shape_1"], job["img_shape_2"]
        total_nfeatures = job["total_nfeatures"]
        output_offset = job["output_offset"]

        patchsize = job["patchsize"]
        nchannels, h, w = shape
        nprows, npcols = h // patchsize, w // patchsize

        job_nfeatures = job["npoints"] + 2
        job_patch_histograms_shape = (nprows, npcols, job_nfeatures)

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job["output_shm_name"])

        all_patch_histograms = np.ndarray(
            (nprows, npcols, total_nfeatures), dtype=_features_dtype, buffer=output_shm.buf
        )
        job_patch_histograms = all_patch_histograms[:, :, output_offset : (output_offset + job_nfeatures)]

        # Try to use cached data
        cached_result_mm = None
        cached_result_mm_pixel = None

        if not tmp_fpath:
            log.debug(f"run_fastlbp: worker {jobname}({pid}): skipping cache")
        else:
            try:
                cached_result_mm = np.load(tmp_fpath, mmap_mode="r")
            except:
                cached_result_mm = None
                log.debug(f"run_fastlbp: worker {jobname}({pid}): no usable cache")

        # read pixel level cache if available
        if not tmp_fpath_pixel:
            log.debug(f"run_fastlbp: worker {jobname}({pid}): skipping pixel cache")
        else:
            try:
                cached_result_mm_pixel = np.load(tmp_fpath_pixel, mmap_mode="r")
            except:
                cached_result_mm_pixel = None
                log.debug(f"run_fastlbp: worker {jobname}({pid}): no usable pixel cache")

        # use cached results if found
        if cached_result_mm is not None:
            # Use cache and return

            log.info(f"run_fastlbp: worker {jobname}({pid}): cache found! copying to output.")
            np.copyto(job_patch_histograms, cached_result_mm)

        # try to read pixel level cache
        elif cached_result_mm_pixel is not None:
            # use pixel level cache to group features into patches and return
            log.info(
                f"run_fastlbp: worker {jobname}({pid}): pixel cache found! Grouping into patches and copying to output."
            )
            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            # TODO: DEBUG: test this
            # add mask/no_mask in the cached result names to distinguish between
            # different types of cache (masked vs not masked)
            if using_patch_mask:
                patch_mask_shm = shared_memory.SharedMemory(name=job["patch_mask_shm_name"])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)

            for pr in range(nprows):
                for pc in range(npcols):
                    if using_patch_mask and patch_mask[pr, pc] == 0:
                        job_patch_histograms[pr, pc, :] = 0
                    else:
                        hist = np.bincount(
                            get_patch(cached_result_mm_pixel, patchsize, pr, pc).flat, minlength=job_nfeatures
                        )
                        job_patch_histograms[pr, pc, :] = hist

            # save grouped cache if it doesn't exists but was requested
            if tmp_fpath:
                log.debug(
                    f"run_chunked_fastlbp: worker {jobname}({pid}): saving cached histograms from the pixel cache"
                )
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_patch_histograms)
                except:
                    log.warning(
                        f"run_chunked_fastlbp: worker {jobname}({pid}): computation successful, but cannot save grouped tmp file from lbp codes"
                    )

        # calculate LBP in two cases
        # 1. No cached results were found
        # 2. Pixel cache was requested but not found, which means it must be created from scratch.
        # (As it is not possible to ungroup the patch cache even if it exists)

        if (cached_result_mm is None and cached_result_mm_pixel is None) or (
            cached_result_mm_pixel is None and tmp_fpath_pixel
        ):
            # Compute full LBP and save different types of cache for now
            # TODO: FIXME: use different flags to save different types of cache
            # (patch level and/or pixel level)

            img_data_shm = shared_memory.SharedMemory(name=job["img_shm_name"])
            img_data = np.ndarray(shape, dtype=job["img_pixel_dtype"], buffer=img_data_shm.buf)

            img_channel = img_data[job["channel"]]
            assert img_channel.flags.c_contiguous
            assert img_channel.dtype == np.uint8

            using_image_mask = "img_mask_shm_name" in job and job["img_mask_shm_name"]
            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            if using_image_mask:
                log.debug(f"run_fastlbp: worker {jobname}({pid}): using image mask")
                img_mask_shm = shared_memory.SharedMemory(name=job["img_mask_shm_name"])
                img_mask = np.ndarray((h, w), dtype=np.uint8, buffer=img_mask_shm.buf)
                lbp_results = uniform_lbp_uint8_masked(
                    image=img_channel, mask=img_mask, P=job["npoints"], R=job["radius"]
                )
                img_mask_shm.close()
            elif using_patch_mask:
                log.debug(f"run_fastlbp: worker {jobname}({pid}): using patch mask")
                patch_mask_shm = shared_memory.SharedMemory(name=job["patch_mask_shm_name"])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)
                lbp_results = uniform_lbp_uint8_patch_masked(
                    image=img_channel, patch_mask=patch_mask, patchsize=patchsize, P=job["npoints"], R=job["radius"]
                )
            else:
                # if no mask is provided
                log.debug(f"run_fastlbp: worker {jobname}({pid}): do not use mask")
                lbp_results = uniform_lbp_uint8(image=img_channel, P=job["npoints"], R=job["radius"])

            # assert lbp_results.dtype == _features_dtype

            img_data_shm.close()

            for pr in range(nprows):
                for pc in range(npcols):
                    if using_patch_mask and patch_mask[pr, pc] == 0:
                        job_patch_histograms[pr, pc, :] = 0
                    else:
                        hist = np.bincount(get_patch(lbp_results, patchsize, pr, pc).flat, minlength=job_nfeatures)
                        job_patch_histograms[pr, pc, :] = hist

            if using_patch_mask:
                patch_mask_shm.close()

            # don't overwrite the grouped cache if it exists and
            # we just recalculated LBP to save raw codes cache
            if tmp_fpath and cached_result_mm is None:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_patch_histograms)
                except:
                    log.warning(
                        f"run_fastlbp: worker {jobname}({pid}): computation successful, but cannot save tmp file"
                    )

            # save raw lbp features as reusable cache for subsequent runs
            if tmp_fpath_pixel:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath_pixel), exist_ok=True)
                    np.save(tmp_fpath_pixel, lbp_results)
                except:
                    log.warning(
                        f"run_fastlbp: worker {jobname}({pid}): computation successful, but cannot save pixel tmp file"
                    )

        output_shm.close()

        log.info(f"run_fastlbp: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")
    except Exception as e:
        log.error(f"run_fastlbp: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    return 0


# prototyping
# TODO: refactor and generalize
def __chunked_worker_fastlbp(df_row_args):

    row_id, job = df_row_args
    tmp_fpath = job["tmp_fpath"]
    tmp_fpath_pixel = job["tmp_fpath_pixel"]

    pid = os.getpid()
    jobname = job["label"]
    log.info(f"run_chunked_fastlbp: worker {pid}: starting job {jobname}")

    try:
        t0 = time.perf_counter()

        # TODO: add support for arbitrary number of axes
        shape = job["img_shape_0"], job["img_shape_1"], job["img_shape_2"]
        total_nfeatures = job["total_nfeatures"]
        output_offset = job["output_offset"]

        patchsize = job["patchsize"]
        chunksize = job["chunksize"]
        padding_radius = job["radius"] + 1

        # TODO: add support for arbitrary number of axes
        chunk_row, chunk_col = job["chunk_origin_0"], job["chunk_origin_1"]  # chunk-wise
        chunk_dim_0, chunk_dim_1 = job["chunk_dim_0"], job["chunk_dim_1"]  # pixel-wise

        nchannels, h, w = shape
        nprows, npcols = h // patchsize, w // patchsize

        # determine the chunk padding in pixels
        padding_top = chunk_row * chunksize * patchsize
        padding_top = padding_radius if padding_radius < padding_top else padding_top

        padding_bottom = h - (chunk_row * chunksize * patchsize + chunk_dim_0)
        padding_bottom = padding_radius if padding_radius < padding_bottom else padding_bottom

        padding_left = chunk_col * chunksize * patchsize
        padding_left = padding_radius if padding_radius < padding_left else padding_left

        padding_right = w - (chunk_col * chunksize * patchsize + chunk_dim_1)
        padding_right = padding_radius if padding_radius < padding_right else padding_right

        # get coordinates and dimensions of the current chunk in patches
        assert chunk_dim_0 % patchsize == 0
        assert chunk_dim_1 % patchsize == 0

        chunk_dim_0_patches = chunk_dim_0 // patchsize
        chunk_dim_1_patches = chunk_dim_1 // patchsize

        chunk_row_in_patches = chunk_row * chunksize
        chunk_col_in_patches = chunk_col * chunksize

        chunk_row_in_pixels = chunk_row_in_patches * patchsize
        chunk_col_in_pixels = chunk_col_in_patches * patchsize

        job_nfeatures = job["npoints"] + 2
        job_patch_histograms_shape = (nprows, npcols, job_nfeatures)

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job["output_shm_name"])

        all_histograms = np.ndarray((nprows, npcols, total_nfeatures), dtype=_features_dtype, buffer=output_shm.buf)

        # each job processes only one chunk

        # get current chunk patches
        chunk_histograms = all_histograms[
            chunk_row_in_patches : (chunk_row_in_patches + chunk_dim_0_patches),
            chunk_col_in_patches : (chunk_col_in_patches + chunk_dim_1_patches),
            :,
        ]

        # get features for the current chunk
        job_chunk_histogram = chunk_histograms[..., output_offset : (output_offset + job_nfeatures)]

        # Try to use cached data
        cached_result_mm = None
        cached_result_mm_pixel = None

        if not tmp_fpath:
            # don't use cache at all
            log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): skipping cache")
        else:
            # try to find existing cached result for the current job
            try:
                cached_result_mm = np.load(tmp_fpath, mmap_mode="r")
            except:
                cached_result_mm = None
                log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): no usable cache")

        # read pixel level cache if available
        if not tmp_fpath_pixel:
            log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): skipping pixel cache")
        else:
            try:
                cached_result_mm_pixel = np.load(tmp_fpath_pixel, mmap_mode="r")
            except:
                cached_result_mm_pixel = None
                log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): no usable pixel cache")

        # use cached results if found
        if cached_result_mm is not None:
            # Use cache and return

            log.info(f"run_chunked_fastlbp: worker {jobname}({pid}): cache found! copying to output.")
            np.copyto(job_chunk_histogram, cached_result_mm)

        # else try to read pixel level cache
        elif cached_result_mm_pixel is not None:
            # use pixel level cache to group features into patches and return
            log.info(
                f"run_chunked_fastlbp: worker {jobname}({pid}): pixel cache found! Grouping into patches and copying to output."
            )

            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            if using_patch_mask:
                patch_mask_shm = shared_memory.SharedMemory(name=job["patch_mask_shm_name"])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)
                patch_mask_chunk = get_padded_region(
                    patch_mask,
                    chunk_row_in_patches,
                    chunk_col_in_patches,
                    chunk_dim_0_patches,
                    chunk_dim_1_patches,
                    0,
                    0,
                    0,
                    0,
                )

            # compute histograms for patches inside the current chunk
            for patch_i in range(chunk_dim_0_patches):
                for patch_j in range(chunk_dim_1_patches):
                    if using_patch_mask and patch_mask_chunk[patch_i, patch_j] == 0:
                        job_chunk_histogram[patch_i, patch_j, :] = 0
                        continue

                    chunk_patch_hist = get_patch(cached_result_mm_pixel, patchsize, patch_i, patch_j)
                    hist = np.bincount(chunk_patch_hist.flat, minlength=job_nfeatures)

                    job_chunk_histogram[patch_i, patch_j, :] = hist

            # save grouped cache if it doesn't exists but was requested
            if tmp_fpath:
                log.debug(
                    f"run_chunked_fastlbp: worker {jobname}({pid}): saving cached histograms from the pixel cache"
                )
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_chunk_histogram)
                except:
                    log.warning(
                        f"run_chunked_fastlbp: worker {jobname}({pid}): computation successful, but cannot save grouped tmp file from lbp codes"
                    )

        # calculate LBP in two cases
        # 1. No cached results were found
        # 2. Pixel cache was requested but not found, which means it must be created from scratch.
        # (As it is not possible to ungroup the patch cache even if it exists)

        if (cached_result_mm is None and cached_result_mm_pixel is None) or (
            cached_result_mm_pixel is None and tmp_fpath_pixel
        ):
            # Compute full LBP **for the current chunk** and save both types of cache

            # TODO: FIXME: use different flags to save different types of cache
            # (patch level and/or pixel level)

            img_data_shm = shared_memory.SharedMemory(name=job["img_shm_name"])
            img_data = np.ndarray(shape, dtype=job["img_pixel_dtype"], buffer=img_data_shm.buf)

            img_channel_full = img_data[job["channel"]]
            assert img_channel_full.flags.c_contiguous
            assert img_channel_full.dtype == np.uint8

            # extract padded chunk from the image to process by the worker
            img_channel_chunk_not_contiguous = get_padded_region(
                img_channel_full,
                chunk_row_in_pixels,
                chunk_col_in_pixels,
                chunk_dim_0,
                chunk_dim_1,
                padding_top,
                padding_bottom,
                padding_left,
                padding_right,
            )

            # TODO: DEBUG:
            # not sure if this is true for the chunk only

            img_channel_chunk = np.ascontiguousarray(img_channel_chunk_not_contiguous)

            assert img_channel_chunk.flags.c_contiguous
            assert img_channel_chunk.dtype == np.uint8

            using_image_mask = "img_mask_shm_name" in job and job["img_mask_shm_name"]
            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            if using_image_mask:
                log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): using image mask")
                img_mask_shm = shared_memory.SharedMemory(name=job["img_mask_shm_name"])
                img_mask = np.ndarray((h, w), dtype=np.uint8, buffer=img_mask_shm.buf)

                # we don't need to use padding for chunk mask as
                # the LBP codes are not computed for the padding region anyway

                # FIXME: reduce mask usage overhead in chunked version
                img_mask_chunk = get_padded_region(
                    img_mask, chunk_row_in_pixels, chunk_col_in_pixels, chunk_dim_0, chunk_dim_1, 0, 0, 0, 0
                )

                img_mask_chunk = np.ascontiguousarray(img_mask_chunk)

                lbp_results = uniform_lbp_uint8_padded_absolute_masked(
                    image=img_channel_chunk,
                    mask=img_mask_chunk,
                    P=job["npoints"],
                    R=job["radius"],
                    abs_r=chunk_row_in_pixels,
                    abs_c=chunk_col_in_pixels,
                    paddings_top_bottom_left_right=[padding_top, padding_bottom, padding_left, padding_right],
                )

                img_mask_shm.close()

            elif using_patch_mask:
                log.debug(f"run_chunked_fastlbp: worker {jobname}({pid}): using patch mask")

                patch_mask_shm = shared_memory.SharedMemory(name=job["patch_mask_shm_name"])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)

                # get region from the patch mask corresponding to the current chunk
                # no padding is needed for that, I just use this func instead of explicit slices
                # TODO: add get_region() function to utils
                patch_mask_chunk = get_padded_region(
                    patch_mask,
                    chunk_row_in_patches,
                    chunk_col_in_patches,
                    chunk_dim_0_patches,
                    chunk_dim_1_patches,
                    0,
                    0,
                    0,
                    0,
                )

                patch_mask_chunk = np.ascontiguousarray(patch_mask_chunk)

                lbp_results = uniform_lbp_uint8_padded_absolute_patch_masked(
                    image=img_channel_chunk,
                    patch_mask=patch_mask_chunk,
                    patchsize=patchsize,
                    P=job["npoints"],
                    R=job["radius"],
                    abs_r=chunk_row_in_pixels,
                    abs_c=chunk_col_in_pixels,
                    paddings_top_bottom_left_right=[padding_top, padding_bottom, padding_left, padding_right],
                )

            else:
                # if no mask is provided
                log.debug(
                    f"run_chunked_fastlbp: worker {jobname}({pid}) absolute coordinates {chunk_row_in_pixels} {chunk_col_in_pixels}: do not use mask"
                )

                lbp_results = uniform_lbp_uint8_padded_absolute(
                    image=img_channel_chunk,
                    P=job["npoints"],
                    R=job["radius"],
                    abs_r=chunk_row_in_pixels,
                    abs_c=chunk_col_in_pixels,
                    paddings_top_bottom_left_right=[padding_top, padding_bottom, padding_left, padding_right],
                )

            # assert lbp_results.dtype == _features_dtype

            img_data_shm.close()

            # group lbp results in the chunk patch-wise

            # compute histograms for patches inside the current chunk
            for patch_i in range(chunk_dim_0_patches):
                for patch_j in range(chunk_dim_1_patches):

                    if using_patch_mask and patch_mask_chunk[patch_i, patch_j] == 0:
                        job_chunk_histogram[patch_i, patch_j, :] = 0
                        continue

                    curr_patch_lbp_results = get_patch(lbp_results, patchsize, patch_i, patch_j)

                    curr_patch_hist = np.bincount(curr_patch_lbp_results.flat, minlength=job_nfeatures)

                    job_chunk_histogram[patch_i, patch_j, :] = curr_patch_hist

            if using_patch_mask:
                patch_mask_shm.close()

            # don't overwrite the grouped cache if it exists and
            # we just recalculated LBP to save raw codes cache
            if tmp_fpath and cached_result_mm is None:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_chunk_histogram)
                except:
                    log.warning(
                        f"run_chunked_fastlbp: worker {jobname}({pid}): computation successful, but cannot save tmp file"
                    )

            # save raw lbp features as reusable cache for subsequent runs
            if tmp_fpath_pixel:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath_pixel), exist_ok=True)
                    np.save(tmp_fpath_pixel, lbp_results)
                except:
                    log.warning(
                        f"run_chunked_fastlbp: worker {jobname}({pid}): computation successful, but cannot save pixel tmp file"
                    )

        output_shm.close()

        log.info(f"run_chunked_fastlbp: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")
    except Exception as e:
        log.error(f"run_chunked_fastlbp: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    return 0


def __single_patch_fastlbp_worker(df_row_args):

    row_id, job = df_row_args

    pid = os.getpid()
    jobname = job["label"]

    return_raw_features = job["return_raw_features"]

    log.info(f"run_patched_fastlbp: worker {pid}: starting job {jobname}")
    try:
        t0 = time.perf_counter()

        # TODO: add support for arbitrary number of axes
        shape = job["img_shape_0"], job["img_shape_1"], job["img_shape_2"]
        total_nfeatures = job["total_nfeatures"]
        output_offset = job["output_offset"]

        patchsize = job["patchsize"]

        left_dist = (patchsize - 1) // 2
        if patchsize % 2 == 0:
            right_dist = patchsize // 2
        else:
            right_dist = left_dist

        top_dist = left_dist
        bottom_dist = right_dist

        padding_radius = job["radius"] + 1

        padding_left = left_dist + padding_radius
        padding_right = right_dist + padding_radius
        padding_top = top_dist + padding_radius
        padding_bottom = bottom_dist + padding_radius

        center_coord_0, center_coord_1 = job["patch_center_coords"]

        nchannels, h, w = shape

        job_nfeatures = job["npoints"] + 2

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job["output_shm_name"])

        all_histograms = np.ndarray(total_nfeatures, dtype=_features_dtype, buffer=output_shm.buf)

        # get features for the job
        job_histogram = all_histograms[output_offset : (output_offset + job_nfeatures)]

        img_data_shm = shared_memory.SharedMemory(name=job["img_shm_name"])
        img_data = np.ndarray(shape, dtype=job["img_pixel_dtype"], buffer=img_data_shm.buf)

        img_channel_full = img_data[job["channel"]]

        assert img_channel_full.flags.c_contiguous
        assert img_channel_full.dtype == np.uint8

        # extract padded patch from the image to process by the worker
        padded_bbox_left, padded_bbox_right = center_coord_1 - padding_left, center_coord_1 + padding_right + 1
        padded_bbox_top, padded_bbox_bottom = center_coord_0 - padding_top, center_coord_0 + padding_bottom + 1

        padded_bbox_left, delta_padding_left = max(padded_bbox_left, 0), (
            0 if padded_bbox_left >= 0 else abs(padded_bbox_left)
        )
        padded_bbox_right, delta_padding_right = min(padded_bbox_right, w), (
            0 if w - padded_bbox_right >= 0 else abs(w - padded_bbox_right)
        )

        padded_bbox_top, delta_padding_top = max(padded_bbox_top, 0), (
            0 if padded_bbox_top >= 0 else abs(padded_bbox_top)
        )
        padded_bbox_bottom, delta_padding_bottom = min(padded_bbox_bottom, h), (
            0 if h - padded_bbox_bottom >= 0 else abs(h - padded_bbox_bottom)
        )

        img_padded_patch = img_channel_full[padded_bbox_top:padded_bbox_bottom, padded_bbox_left:padded_bbox_right]

        img_padded_patch_contiguous = np.ascontiguousarray(img_padded_patch)

        assert img_padded_patch_contiguous.flags.c_contiguous
        assert img_padded_patch_contiguous.dtype == np.uint8

        # if no mask is provided
        log.debug(f"run_patched_fastlbp: worker {jobname}({pid}) patch coordinates {center_coord_0} {center_coord_1}")

        lbp_results = uniform_lbp_uint8_padded_absolute(
            image=img_padded_patch_contiguous,
            P=job["npoints"],
            R=job["radius"],
            abs_r=max(center_coord_1 - padding_top, 0),
            abs_c=max(center_coord_0 - padding_left, 0),
            paddings_top_bottom_left_right=[
                padding_radius - delta_padding_top,
                padding_radius - delta_padding_bottom,
                padding_radius - delta_padding_left,
                padding_radius - delta_padding_right,
            ],
        )

        assert lbp_results.shape == (patchsize, patchsize)

        img_padded_patch_to_return = img_padded_patch.copy()

        img_data_shm.close()

        job_histogram[:] = np.bincount(lbp_results.flat, minlength=job_nfeatures)

        output_shm.close()

        log.info(f"run_patched_fastlbp: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")

    except Exception as e:
        log.error(f"run_patched_fastlbp: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    if return_raw_features:
        return lbp_results
    else:
        return 0


def __cuda_worker_fastlbp(df_row_args):
    if not _HAS_CUDA_EXT:
        raise ImportError("fastlbp GPU extension not available. Install with CUDA and the [gpu] extra.")

    row_id, job = df_row_args
    tmp_fpath = job["tmp_fpath"]
    tmp_fpath_pixel = job["tmp_fpath_pixel"]

    pid = os.getpid()
    jobname = job["label"]
    log.info(f"run_cuda_fastlbp: worker {pid}: starting job {jobname}")

    try:
        t0 = time.perf_counter()

        # TODO: add support for arbitrary number of axes
        shape = job["img_shape_0"], job["img_shape_1"], job["img_shape_2"]
        total_nfeatures = job["total_nfeatures"]
        total_raw_nfeatures = job["total_raw_nfeatures"]
        output_offset = job["output_offset"]
        raw_output_dimension = job["raw_output_dimension"]

        patchsize = job["patchsize"]
        chunksize = job["chunksize"]
        padding_radius = job["radius"] + 1

        # TODO: add support for arbitrary number of axes
        chunk_row, chunk_col = job["chunk_origin_0"], job["chunk_origin_1"]  # chunk-wise
        chunk_dim_0, chunk_dim_1 = job["chunk_dim_0"], job["chunk_dim_1"]  # pixel-wise

        nchannels, h, w = shape
        nprows, npcols = h // patchsize, w // patchsize

        # determine the chunk padding in pixels
        padding_top = chunk_row * chunksize * patchsize
        padding_top = padding_radius if padding_radius < padding_top else padding_top

        padding_bottom = h - (chunk_row * chunksize * patchsize + chunk_dim_0)
        padding_bottom = padding_radius if padding_radius < padding_bottom else padding_bottom

        padding_left = chunk_col * chunksize * patchsize
        padding_left = padding_radius if padding_radius < padding_left else padding_left

        padding_right = w - (chunk_col * chunksize * patchsize + chunk_dim_1)
        padding_right = padding_radius if padding_radius < padding_right else padding_right

        # get coordinates and dimensions of the current chunk in patches
        # print(f'{chunk_dim_0 = } DEBUG')
        assert chunk_dim_0 % patchsize == 0
        assert chunk_dim_1 % patchsize == 0

        chunk_dim_0_patches = chunk_dim_0 // patchsize
        chunk_dim_1_patches = chunk_dim_1 // patchsize

        chunk_row_in_patches = chunk_row * chunksize
        chunk_col_in_patches = chunk_col * chunksize

        chunk_row_in_pixels = chunk_row_in_patches * patchsize
        chunk_col_in_pixels = chunk_col_in_patches * patchsize

        job_nfeatures = job["npoints"] + 2
        job_patch_histograms_shape = (nprows, npcols, job_nfeatures)

        # Obtain output memory
        output_shm = shared_memory.SharedMemory(name=job["output_shm_name"])

        raw_output_shm = None
        if job["raw_output_shm_name"] is not None:
            raw_output_shm = shared_memory.SharedMemory(name=job["raw_output_shm_name"])

        all_histograms = np.ndarray((nprows, npcols, total_nfeatures), dtype=_features_dtype, buffer=output_shm.buf)

        if raw_output_shm is not None:
            all_raw_features = np.ndarray(
                (h, w, total_raw_nfeatures), dtype=_raw_features_dtype, buffer=raw_output_shm.buf
            )
        else:
            all_raw_features = None

        # each job processes only one chunk

        # get current chunk patches
        # print(f'{chunk_row_in_patches = } {chunk_dim_0_patches = } worker debug')
        chunk_histograms = all_histograms[
            chunk_row_in_patches : (chunk_row_in_patches + chunk_dim_0_patches),
            chunk_col_in_patches : (chunk_col_in_patches + chunk_dim_1_patches),
            :,
        ]

        chunk_raw_features = (
            all_raw_features[
                (chunk_row_in_pixels - padding_top) : (chunk_row_in_pixels + chunk_dim_0 + padding_bottom),
                (chunk_col_in_pixels - padding_left) : (chunk_col_in_pixels + chunk_dim_1 + padding_right),
                :,
            ]
            if all_raw_features is not None
            else None
        )

        # print(f'{all_raw_features.shape = } {chunk_raw_features.shape = }') DEBUG

        # get features for the current chunk
        job_chunk_histogram = chunk_histograms[..., output_offset : (output_offset + job_nfeatures)]
        job_chunk_raw_features = (
            chunk_raw_features[..., raw_output_dimension] if chunk_raw_features is not None else None
        )

        # Try to use cached data
        cached_result_mm = None
        cached_result_mm_pixel = None

        if not tmp_fpath:
            # don't use cache at all
            log.debug(f"run_cuda_fastlbp: worker {jobname}({pid}): skipping cache")
        else:
            # try to find existing cached result for the current job
            try:
                cached_result_mm = np.load(tmp_fpath, mmap_mode="r")
            except:
                cached_result_mm = None
                log.debug(f"run_cuda_fastlbp: worker {jobname}({pid}): no usable cache")

        # read pixel level cache if available
        if not tmp_fpath_pixel:
            log.debug(f"run_cuda_fastlbp: worker {jobname}({pid}): skipping pixel cache")
        else:
            try:
                cached_result_mm_pixel = np.load(tmp_fpath_pixel, mmap_mode="r")
            except:
                cached_result_mm_pixel = None
                log.debug(f"run_cuda_fastlbp: worker {jobname}({pid}): no usable pixel cache")

        # use cached results if found
        if cached_result_mm is not None:
            # Use cache and return

            log.info(f"run_cuda_fastlbp: worker {jobname}({pid}): cache found! copying to output.")
            np.copyto(job_chunk_histogram, cached_result_mm)

        # else try to read pixel level cache
        elif cached_result_mm_pixel is not None:
            # use pixel level cache to group features into patches and return
            log.info(
                f"run_cuda_fastlbp: worker {jobname}({pid}): pixel cache found! Grouping into patches and copying to output."
            )

            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            if using_patch_mask:
                patch_mask_shm = shared_memory.SharedMemory(name=job["patch_mask_shm_name"])
                patch_mask = np.ndarray((nprows, npcols), dtype=np.uint8, buffer=patch_mask_shm.buf)
                patch_mask_chunk = get_padded_region(
                    patch_mask,
                    chunk_row_in_patches,
                    chunk_col_in_patches,
                    chunk_dim_0_patches,
                    chunk_dim_1_patches,
                    0,
                    0,
                    0,
                    0,
                )

            # compute histograms for patches inside the current chunk
            for patch_i in range(chunk_dim_0_patches):
                for patch_j in range(chunk_dim_1_patches):
                    if using_patch_mask and patch_mask_chunk[patch_i, patch_j] == 0:
                        job_chunk_histogram[patch_i, patch_j, :] = 0
                        continue

                    chunk_patch_hist = get_patch(cached_result_mm_pixel, patchsize, patch_i, patch_j)
                    hist = np.bincount(chunk_patch_hist.flat, minlength=job_nfeatures)

                    job_chunk_histogram[patch_i, patch_j, :] = hist

            # save grouped cache if it doesn't exists but was requested
            if tmp_fpath:
                log.debug(f"run_cuda_fastlbp: worker {jobname}({pid}): saving cached histograms from the pixel cache")
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_chunk_histogram)
                except:
                    log.warning(
                        f"run_cuda_fastlbp: worker {jobname}({pid}): computation successful, but cannot save grouped tmp file from lbp codes"
                    )

        # calculate LBP in two cases
        # 1. No cached results were found
        # 2. Pixel cache was requested but not found, which means it must be created from scratch.
        # (As it is not possible to ungroup the patch cache even if it exists)

        if (cached_result_mm is None and cached_result_mm_pixel is None) or (
            cached_result_mm_pixel is None and tmp_fpath_pixel
        ):
            # Compute full LBP **for the current chunk** and save both types of cache

            # TODO: FIXME: use different flags to save different types of cache
            # (patch level and/or pixel level)

            img_data_shm = shared_memory.SharedMemory(name=job["img_shm_name"])
            img_data = np.ndarray(shape, dtype=job["img_pixel_dtype"], buffer=img_data_shm.buf)

            img_channel_full = img_data[job["channel"]]
            assert img_channel_full.flags.c_contiguous
            assert img_channel_full.dtype == np.uint8

            # extract padded chunk from the image to process by the worker
            img_channel_chunk_not_contiguous = get_padded_region(
                img_channel_full,
                chunk_row_in_pixels,
                chunk_col_in_pixels,
                chunk_dim_0,
                chunk_dim_1,
                padding_top,
                padding_bottom,
                padding_left,
                padding_right,
            )

            # print(f'{img_channel_chunk_not_contiguous.shape = } {(chunk_dim_0, chunk_dim_1) = }') DEBUG

            # TODO: DEBUG:
            # not sure if this is true for the chunk only

            img_channel_chunk = np.ascontiguousarray(img_channel_chunk_not_contiguous)

            # print(f'{chunk_row_in_pixels = } {chunk_col_in_pixels} {chunk_dim_0} {chunk_dim_1}')
            # print(f'{padding_top = } {padding_bottom = } {padding_left = } {padding_right = }')
            # print(f'{img_channel_chunk_not_contiguous.shape = } {img_channel_chunk.shape = }')

            assert img_channel_chunk.flags.c_contiguous
            assert img_channel_chunk.dtype == np.uint8

            using_image_mask = "img_mask_shm_name" in job and job["img_mask_shm_name"]
            using_patch_mask = "patch_mask_shm_name" in job and job["patch_mask_shm_name"]

            if using_image_mask:
                # TODO:
                return

            elif using_patch_mask:
                # TODO:
                return

            else:
                # if no mask is provided
                log.debug(
                    f"run_cuda_fastlbp: worker {jobname}({pid}) absolute coordinates {chunk_row_in_pixels} {chunk_col_in_pixels}: do not use mask"
                )

                lbp_results = np.zeros(shape=img_channel_chunk.shape, dtype=np.uint32)

                cuda_lbp(
                    img_channel_chunk, lbp_results, P=job["npoints"], R=job["radius"]
                )  # FIXME TODO: use absolute coordinates as in the CPU chunked version!!
                lbp_results = lbp_results.astype(np.uint16)  # DEBUG TODO: remove this

            # assert lbp_results.dtype == _features_dtype

            img_data_shm.close()

            # group lbp results in the chunk patch-wise
            # print(f'{job_chunk_raw_features.shape = } {lbp_results.shape = } {(padding_top, padding_bottom, padding_left, padding_right) = }') DEBUG
            if job_chunk_raw_features is not None:
                np.copyto(job_chunk_raw_features, lbp_results)

            # compute histograms for patches inside the current chunk
            for patch_i in range(chunk_dim_0_patches):
                for patch_j in range(chunk_dim_1_patches):

                    if using_patch_mask and patch_mask_chunk[patch_i, patch_j] == 0:
                        job_chunk_histogram[patch_i, patch_j, :] = 0
                        continue

                    curr_patch_lbp_results = get_patch(lbp_results, patchsize, patch_i, patch_j)

                    curr_patch_hist = np.bincount(curr_patch_lbp_results.flat, minlength=job_nfeatures)

                    job_chunk_histogram[patch_i, patch_j, :] = curr_patch_hist

            if using_patch_mask:
                patch_mask_shm.close()

            # don't overwrite the grouped cache if it exists and
            # we just recalculated LBP to save raw codes cache
            if tmp_fpath and cached_result_mm is None:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath), exist_ok=True)
                    np.save(tmp_fpath, job_chunk_histogram)
                except:
                    log.warning(
                        f"run_cuda_fastlbp: worker {jobname}({pid}): computation successful, but cannot save tmp file"
                    )

            # save raw lbp features as reusable cache for subsequent runs
            if tmp_fpath_pixel:
                try:
                    os.makedirs(os.path.dirname(tmp_fpath_pixel), exist_ok=True)
                    np.save(tmp_fpath_pixel, lbp_results)
                except:
                    log.warning(
                        f"run_cuda_fastlbp: worker {jobname}({pid}): computation successful, but cannot save pixel tmp file"
                    )

        output_shm.close()

        log.info(f"run_cuda_fastlbp: worker {pid}: finished job {jobname} in {time.perf_counter()-t0:.5g}s")
    except Exception as e:
        log.error(f"run_cuda_fastlbp: worker {jobname}({pid}): exception! Aborting execution.")
        log.error(e, exc_info=True)

    return 0
