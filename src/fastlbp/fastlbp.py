import logging
import os
import time
from collections import namedtuple
from multiprocessing import Pool, shared_memory
from typing import Iterable, Literal, Union

import numpy as np
import pandas as pd
import psutil
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame

from .common import _features_dtype
from .utils import int_verbosity_to_logger_level, patchify_image_mask
from .workers import (
    __chunked_worker_fastlbp,
    __single_patch_fastlbp_worker,
    __worker_fastlbp,
)

logging.basicConfig()
log = logging.getLogger("fastlbp")

DEFAULT_LEVEL = logging.WARNING

log.setLevel(DEFAULT_LEVEL)

#####
# MISC ROUTINES FOR INTERNAL USAGE


def __create_pipeline_hash(method_name, *pipeline_params):
    import hashlib

    from . import __version__

    s = __version__ + ";" + method_name + ";" + (";".join([str(p) for p in pipeline_params]))
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:7]


def __sanitize_img_name(img_name):
    return img_name.replace(".", "_").replace("-", "_")


def __sanitize_outfile_name(outfile_name):
    if outfile_name.endswith(".npy"):
        return outfile_name
    return outfile_name + ".npy"


def __get_output_dir(path: str | None = None):
    return "data/out" if path is None else path


def __get_tmp_dir(pipeline_name, path: str | None = None):
    tmp_savedir = "data/tmp/" if path is None else path
    return os.path.join(tmp_savedir, pipeline_name)


def __get_tmp_dir_explicit(pipeline_name, path: str):
    return os.path.join(path, pipeline_name)


def is_cuda_available() -> bool:
    """Return True if the compiled CUDA extension is importable."""
    try:
        from ._lbp_gpu import cuda_lbp  # noqa: F401

        return True
    except Exception:
        return False


def __check_existing_result_on_disk(
    outdir: str, outfile_name: str, caller: str = "run_fastlbp", pipeline_hash: str = "", overwrite_output: bool = False
):
    res_outdir = outdir if outdir else os.getcwd()
    output_fpath = os.path.join(res_outdir, outfile_name)
    output_abspath = os.path.abspath(output_fpath)
    try:
        if os.path.exists(output_fpath) and not overwrite_output:
            log.error(
                f"{caller}({pipeline_hash}): overwrite_output is False and output file {output_abspath} already exists. Aborting."
            )
            return FastlbpResult(output_abspath, None)

        os.makedirs(res_outdir, exist_ok=True)
        if not os.access(res_outdir, os.W_OK):
            log.error(
                f"{caller}({pipeline_hash}): output dir {os.path.dirname(output_abspath)} is not writable. Aborting."
            )
            return FastlbpResult(output_abspath, None)
    except:
        log.error(f"{caller}({pipeline_hash}): error accessing output dir {os.path.dirname(output_abspath)}. Aborting.")
        return FastlbpResult(output_abspath, None)

    return None


def __create_patch_mask_shm(
    nprows: int,
    npcols: int,
    patchsize: int,
    mask_method: str,
    pipeline_hash: str,
    img_mask: np.ndarray = None,
    img_patch_mask: np.ndarray = None,
    caller: str = "run_fastlbp",
):
    """
    We won't compute a patch if it has at least one pixel is ignored in img_mask.
    EVERY feature for this patch will be zero.
    Thus it is sensible to store a patch-wise mask, not the whole image mask.
    This behavior might change in the future.

    e.g. 'exclude whole patch if at least 1 zero' vs 'include whole patch if at least 1 non-zero'
    """

    patch_mask_shm = None  # per patch mask
    patch_mask_shape = (nprows, npcols)

    if img_mask is not None:
        log.info(f"{caller}({pipeline_hash}): using image mask.")
        patch_mask = patchify_image_mask(img_mask, patchsize, edit_img_mask=False, method=mask_method)
        assert patch_mask.shape == patch_mask_shape

        patch_mask_shm = shared_memory.SharedMemory(create=True, size=patch_mask.nbytes)
        patch_mask_np = np.ndarray(patch_mask_shape, dtype=np.uint8, buffer=patch_mask_shm.buf)
        np.copyto(patch_mask_np, patch_mask, casting="no")

        log.info(
            f"{caller}({pipeline_hash}): pixel mask converted to patch mask. Created shared memory for patch mask."
        )

        return patch_mask, patch_mask_shm

    elif img_patch_mask is not None:
        log.info(f"{caller}({pipeline_hash}): using patch mask.")
        patch_mask = img_patch_mask

        assert patch_mask.shape == patch_mask_shape

        patch_mask_shm = shared_memory.SharedMemory(create=True, size=patch_mask.nbytes)
        patch_mask_np = np.ndarray(patch_mask_shape, dtype=np.uint8, buffer=patch_mask_shm.buf)
        np.copyto(patch_mask_np, patch_mask, casting="no")

        log.info(f"{caller}({pipeline_hash}): patch mask copied into shared memory region.")

        return patch_mask, patch_mask_shm

    return None


def __log_fastlbp_setup(caller: str, **setup_kwargs):

    passed_args_str = " ".join([f"{k}={v} " for k, v in setup_kwargs.items()])

    log.info(f"{caller}: params:")
    if "savefile" in setup_kwargs.keys():
        log.info("fastLBP is running in disk mode")
    log.info(f"{passed_args_str}")


def __get_optimal_ncpus(ncpus: int = -1) -> None:

    assert ncpus >= -1
    max_ncpus = psutil.cpu_count(logical=False)

    if ncpus > max_ncpus:
        log.warning(
            f"ncpus ({ncpus}) greater than number of physical cpus ({max_ncpus})! Beware the performance issues."
        )

    if ncpus == -1:
        log.info(f"ncpus == -1 so using all available physical cpus. That is, {max_ncpus} processes")
        ncpus = max_ncpus

    return ncpus


def __get_optimal_ram(max_ram=None) -> None:

    if max_ram is not None:
        log.warning("max_ram parameter is currently ignored!")

    return max_ram


# TODO: test if works as intended
def __unlink_and_close_shms(*shms_to_cleanup) -> None:
    for shm in shms_to_cleanup:
        if shm is not None:
            shm.unlink()
            shm.close()


def __generate_jobs_dataframe(
    index_params: list, index_names: list[str], fastlbp_method: str = "standard"
) -> pd.DataFrame:

    jobs_index = pd.MultiIndex.from_product(index_params, names=index_names)

    # create a list of jobs
    jobs = pd.DataFrame(
        index=jobs_index,
        columns=[
            "channel",
            "radius",
            "img_name",
            "label",
            "npoints",
            "patchsize",
            "img_shm_name",
            "img_pixel_dtype",
            "img_shape_0",
            "img_shape_1",
            "img_shape_2",
            "output_shm_name",
            "output_offset",
        ],
    )  # essential columns used in all fastlbp versions

    return jobs


def __fill_in_channel_features_offset_in_jobs_df(
    jobs: pd.DataFrame, channel_list: list, columns_to_replicate_for_channels: dict, nfeatures_cumsum: np.ndarray
) -> None:
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_output_offset = 0
    for c in channel_list:
        jobs.loc[c, "channel"] = c

        for col_name, col_val in columns_to_replicate_for_channels.items():
            jobs.loc[c, col_name] = col_val

        jobs.loc[c, "output_offset"] = channel_output_offset + np.hstack([[0], nfeatures_cumsum[:-1]])
        channel_output_offset += nfeatures_per_channel


def __fill_in_cache_dir_in_jobs_df(
    jobs: pd.DataFrame, pipeline_name: str, cache_dir: str, job_label_colname: str, colname: str
) -> None:

    assert job_label_colname in jobs.columns

    # if cache_dir is set, save cached lbp results
    if cache_dir is not None:
        base_tmp_path = __get_tmp_dir_explicit(pipeline_name, cache_dir)
        jobs[colname] = jobs.apply(
            lambda row: os.path.join(base_tmp_path, row[job_label_colname]) + ".npy", axis="columns"
        )
    else:
        jobs[colname] = ""


def __fill_in_cache_dirs_in_jobs_df(
    jobs: pd.DataFrame, pipeline_name: str, cache_dirs: Iterable, jobs_label_colnames: Iterable, colnames: Iterable
) -> None:

    assert len(cache_dirs) == len(jobs_label_colnames) == len(colnames)

    for idx, cache_dir in enumerate(cache_dirs):
        job_label_colname = jobs_label_colnames[idx]
        colname = colnames[idx]
        __fill_in_cache_dir_in_jobs_df(jobs, pipeline_name, cache_dir, job_label_colname, colname)


def __dump_jobs_df_to_csv(jobs: pd.DataFrame, jobs_csv_savefile: str) -> None:
    jobs_csv_savedir = os.path.dirname(jobs_csv_savefile)
    if jobs_csv_savedir:
        os.makedirs(jobs_csv_savedir, exist_ok=True)
    jobs.to_csv(jobs_csv_savefile)


## chunked fastlbp helper functions


def __get_chunk_origins_and_shapes(
    nprows: int, npcols: int, patchsize: int, chunksize: int
) -> tuple[int, int, int, int]:
    n_chunk_rows = nprows // chunksize
    n_chunk_cols = npcols // chunksize
    row_chunk_indices, col_chunk_indices = np.arange(n_chunk_rows), np.arange(n_chunk_cols)

    row_chunk_dims = np.full(n_chunk_rows, patchsize * chunksize)
    col_chunk_dims = np.full(n_chunk_cols, patchsize * chunksize)

    remaining_pixels_rows = (nprows % chunksize) * patchsize
    if remaining_pixels_rows > 0:
        row_chunk_indices = np.append(row_chunk_indices, [n_chunk_rows])
        row_chunk_dims = np.append(row_chunk_dims, [remaining_pixels_rows])

    remaining_pixels_cols = (npcols % chunksize) * patchsize
    if remaining_pixels_cols > 0:
        col_chunk_indices = np.append(col_chunk_indices, [n_chunk_cols])
        col_chunk_dims = np.append(col_chunk_dims, [remaining_pixels_cols])

    return row_chunk_indices, col_chunk_indices, row_chunk_dims, col_chunk_dims


#####
# PUBLIC UTILS


def get_radii(n: int = 15) -> list[float]:
    """
    Get a standard sequence of radii.

    The formula is `round(1.499*1.327**(float(x)))`.
    It was coined by Ben in his initial lbp pipeline.
    """
    radius_list = [round(1.499 * 1.327 ** (float(x))) for x in range(0, n)]
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
    return np.ceil(2 * np.pi * r).astype("int")


#####
# PIPELINE METHODS

FastlbpResult = namedtuple("FastlbpResult", "output_abspath, patch_mask")


def run_fastlbp(
    img_data: ArrayLike,
    radii_list: ArrayLike,
    npoints_list: ArrayLike,
    patchsize: int,
    ncpus: int,
    img_mask=None,
    img_patch_mask=None,
    mask_method="any",
    max_ram=None,
    img_name="img",
    savefile: str = "",
    overwrite_output: bool = False,
    jobs_csv_savefile: str | None = None,
    histograms_cache_dir: str | None = None,
    lbp_codes_cache_dir: str | None = None,
    verbosity: int = 1,
) -> Union[FastlbpResult, np.array]:
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

    `img_name_pixel_cache`: default "img_pixel_cache". Name to use for LBP cache before grouping into patches.
    Thereby the name must not contain the patch size in for correct caching across multiple patch sizes.

    `outfile_name`: default 'lbp_features.npy'. Name of an output file. The final path is `'./data/out/{outfile_name}.npy'`.
    You cannot change the path yet, I am sorry :(

    `save_intermediate_results`: bool | str, default True. Path to the cached results folder. `True` will save the cache in the working dir. This is espetially useful if computation
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

    logger_level = int_verbosity_to_logger_level(verbosity)
    log.setLevel(logger_level)

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) in [2, 3]
    assert img_data.dtype == np.uint8

    if len(img_data.shape) == 2:
        img_data = img_data[:, :, None]

    if img_mask is not None:
        assert img_mask.shape == img_data.shape[:2]
        assert img_mask.dtype == np.uint8

    t = time.perf_counter()

    log.info("run_fastlbp: initial setup...")

    img_name = __sanitize_img_name(img_name)
    outdir, outfile_name = os.path.dirname(savefile), os.path.basename(savefile)

    if outfile_name:
        outfile_name = __sanitize_outfile_name(outfile_name)

    # data_hash = hashlib.sha1(img_data.data).hexdigest()

    # this way pipelines with different ncpus/radii/npoints can reuse tmp files if patchsize, img name and version are the same
    pipeline_hash = __create_pipeline_hash(
        "fastlbp",
        [
            str(img_data.shape),
            patchsize,
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    # pixel cache consists of LBP codes before they are grouped into histograms
    # therefore it is invariant to the patch size parameter and
    # requires a separate hash that does not depend on patch size
    pipeline_hash_lbp_codes = __create_pipeline_hash(
        "fastlbp-chunked",
        [
            str(img_data.shape),
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    pipeline_name = f"{img_name}-fastlbp-{pipeline_hash}"
    pipeline_name_lbp_codes = f"{img_name}-fastlbp-chunked-{pipeline_hash_lbp_codes}"

    __log_fastlbp_setup(
        caller="run_fastlbp",
        img_data_shape=img_data.shape,
        radii_list=radii_list,
        npoints_list=npoints_list,
        patchsize=patchsize,
        ncpus=ncpus,
        max_ram=max_ram,
        img_name=img_name,
        histograms_cache_dir=histograms_cache_dir,
        lbp_codes_cache_dir=lbp_codes_cache_dir,
        savefile=savefile,
        pipeline_hash=pipeline_hash,
        pipeline_hash_lbp_codes=pipeline_hash_lbp_codes,
    )

    ncpus = __get_optimal_ncpus(ncpus)
    max_ram = __get_optimal_ram(max_ram)

    if savefile:
        calculated_lbp = __check_existing_result_on_disk(
            outdir, outfile_name, caller="run_fastlbp", pipeline_hash=pipeline_hash, overwrite_output=overwrite_output
        )
        if calculated_lbp is not None:
            return calculated_lbp

    log.info(f"run_fastlbp({pipeline_hash}): initial setup took {time.perf_counter()-t:.5g}s")
    log.info(f"run_fastlbp({pipeline_hash}): creating a list of jobs...")

    t = time.perf_counter()

    # method-specific params

    h, w, nchannels = img_data.shape
    nprows, npcols = h // patchsize, w // patchsize
    nfeatures_cumsum = np.cumsum(np.array(npoints_list) + 2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    # create a list of jobs
    jobs = __generate_jobs_dataframe(index_params=[channel_list, radii_list], index_names=["channel", "radius"])

    jobs["img_name"] = img_name

    __fill_in_channel_features_offset_in_jobs_df(
        jobs, channel_list, {"radius": radii_list, "npoints": npoints_list}, nfeatures_cumsum
    )

    jobs["label"] = jobs.apply(
        lambda row: f"{img_name}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}", axis="columns"
    )  # channel and radius are in index

    img_name_pixel_cache = img_name + "_lbp_codes"
    jobs["pixel_cache_label"] = jobs.apply(
        lambda row: f"{img_name_pixel_cache}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}", axis="columns"
    )  # channel and radius are in index

    jobs["patchsize"] = patchsize

    __fill_in_cache_dirs_in_jobs_df(
        jobs,
        pipeline_name,
        [histograms_cache_dir, lbp_codes_cache_dir],
        ["label", "pixel_cache_label"],
        ["tmp_fpath", "tmp_fpath_pixel"],
    )

    total_nfeatures = nfeatures_per_channel * len(channel_list)
    patch_features_shape = (nprows, npcols, total_nfeatures)
    jobs["total_nfeatures"] = total_nfeatures

    # Prepare contiguous array.
    # Channels will go first. Then h and w.
    img_data = np.ascontiguousarray(np.moveaxis(img_data, (0, 1, 2), (1, 2, 0)))

    log.info(f"run_fastlbp({pipeline_hash}): creating shared memory")
    # create shared memory for input image
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)

    # copy image to shared memory
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    np.copyto(input_img_np, img_data, casting="no")

    # copy mask to shared memory if provided.

    patch_mask_shm_result = __create_patch_mask_shm(
        nprows, npcols, patchsize, mask_method, pipeline_hash, img_mask, img_patch_mask, caller="run_fastlbp"
    )

    patch_mask = None
    patch_mask_shm = None

    if patch_mask_shm_result is not None:
        patch_mask, patch_mask_shm = patch_mask_shm_result

    # create and initialize shared memory for output
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype(_features_dtype).itemsize)
    )
    patch_features = np.ndarray(patch_features_shape, _features_dtype, buffer=patch_features_shm.buf)

    patch_features.fill(0)
    log.info(f"run_fastlbp({pipeline_hash}): shared memory created")

    jobs["img_shm_name"] = input_img_shm.name
    jobs["patch_mask_shm_name"] = patch_mask_shm.name if patch_mask_shm is not None else ""
    jobs["img_pixel_dtype"] = input_img_np.dtype  # note: always uint8
    jobs["img_shape_0"] = input_img_np.shape[0]  # nchannels
    jobs["img_shape_1"] = input_img_np.shape[1]  # h
    jobs["img_shape_2"] = input_img_np.shape[2]  # w
    jobs["output_shm_name"] = patch_features_shm.name

    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    if jobs_csv_savefile is not None:
        __dump_jobs_df_to_csv(jobs, jobs_csv_savefile)

    log.info(f"run_fastlbp({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s")
    log.info(f"run_fastlbp({pipeline_hash}): jobs:")
    log.info(jobs.head())
    log.info(f"Jobs DataFrame shape: {jobs.shape}")

    assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f"run_fastlbp({pipeline_hash}): start computation")
    t0 = time.perf_counter()
    with Pool(ncpus) as pool:
        jobs_results = pool.map(func=__worker_fastlbp, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f"run_fastlbp({pipeline_hash}): computation finished in {t_elapsed:.5g}s. Start saving")

    # save results
    lbp_result = None
    output_fpath = os.path.join(outdir, outfile_name)
    output_abspath = os.path.abspath(output_fpath)

    if savefile:
        np.save(output_fpath, patch_features)
    else:
        lbp_result = patch_features.copy()

    __unlink_and_close_shms(input_img_shm, patch_features_shm, patch_mask_shm)

    log.info(f"run_fastlbp({pipeline_hash}): shared memory unlinked. Goodbye")

    # reset logger to its original level
    log.setLevel(DEFAULT_LEVEL)

    if savefile:
        return FastlbpResult(output_abspath, patch_mask)
    else:
        return lbp_result


def run_chunked_fastlbp(
    img_data: ArrayLike,
    radii_list: ArrayLike,
    npoints_list: ArrayLike,
    patchsize: int,
    ncpus: int,
    chunksize: int = 20,
    img_mask=None,
    img_patch_mask=None,
    mask_method="any",
    max_ram=None,
    img_name="img_chunked",
    savefile: str = "",
    overwrite_output: bool = False,
    jobs_csv_savefile: str | None = None,
    histograms_cache_dir: str | None = None,
    lbp_codes_cache_dir: str | None = None,
    verbosity: int = 1,
) -> Union[FastlbpResult, np.array]:
    """
    The main idea is to split the image into overlapping chunks that enclose
    patches with a surrounding padding of the largest radius size.

    verbosity: int, default 1.
        Choose between 1, 2, 3, 4.
    """

    logger_level = int_verbosity_to_logger_level(verbosity)
    log.setLevel(logger_level)

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) in [2, 3]
    assert img_data.dtype == np.uint8

    if len(img_data.shape) == 2:
        img_data = img_data[:, :, None]

    if img_mask is not None:
        assert img_mask.shape == img_data.shape[:2]
        assert img_mask.dtype == np.uint8

    t = time.perf_counter()

    log.info("run_chunked_fastlbp: initial setup...")

    img_name = __sanitize_img_name(img_name)
    outdir, outfile_name = os.path.dirname(savefile), os.path.basename(savefile)

    if outfile_name:
        outfile_name = __sanitize_outfile_name(outfile_name)

    # this way pipelines with different ncpus/radii/npoints can reuse tmp files if patchsize, img name and version are the same
    pipeline_hash = __create_pipeline_hash(
        "fastlbp-chunked",
        [
            str(img_data.shape),
            patchsize,
            chunksize,
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    # pixel cache consists of LBP codes before they are grouped into histograms
    # therefore it is invariant to the patch size parameter and
    # requires a separate hash that does not depend on patch size

    # UPDATE: the statement above is not correct for the chunked version
    # the chunks consist of a whole number of patches
    # meaning that changing the patch size will alter the splitting pattern
    # and the shape of the cached results
    # TODO: add separate pipeline hash for pixel cache in the regular function (not chunked)
    pipeline_hash_lbp_codes = __create_pipeline_hash(
        "fastlbp-chunked",
        [
            str(img_data.shape),
            patchsize,
            chunksize,
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    pipeline_name = f"{img_name}-fastlbp-chunked-{pipeline_hash}"
    pipeline_name_lbp_codes = f"{img_name}-fastlbp-chunked-{pipeline_hash_lbp_codes}"

    __log_fastlbp_setup(
        caller="run_chunked_fastlbp",
        img_data_shape=img_data.shape,
        radii_list=radii_list,
        npoints_list=npoints_list,
        patchsize=patchsize,
        chunksize=chunksize,
        ncpus=ncpus,
        max_ram=max_ram,
        img_name=img_name,
        histograms_cache_dir=histograms_cache_dir,
        lbp_codes_cache_dir=lbp_codes_cache_dir,
        savefile=savefile,
        pipeline_hash=pipeline_hash,
        pipeline_hash_lbp_codes=pipeline_hash_lbp_codes,
    )

    ncpus = __get_optimal_ncpus(ncpus)
    max_ram = __get_optimal_ram(max_ram)

    if savefile:
        calculated_lbp = __check_existing_result_on_disk(
            outdir,
            outfile_name,
            caller="run_chunked_fastlbp",
            pipeline_hash=pipeline_hash,
            overwrite_output=overwrite_output,
        )
        if calculated_lbp is not None:
            return calculated_lbp

    log.info(f"run_chunked_fastlbp({pipeline_hash}): initial setup took {time.perf_counter()-t:.5g}s")
    log.info(f"run_chunked_fastlbp({pipeline_hash}): creating a list of jobs...")

    t = time.perf_counter()

    # method-specific params

    h, w, nchannels = img_data.shape
    nprows, npcols = h // patchsize, w // patchsize
    nfeatures_cumsum = np.cumsum(np.array(npoints_list) + 2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    # get chunk origins and shapes

    row_chunk_indices, col_chunk_indices, row_chunk_dims, col_chunk_dims = __get_chunk_origins_and_shapes(
        nprows, npcols, patchsize, chunksize
    )

    assert len(row_chunk_indices) == len(row_chunk_dims)
    assert len(col_chunk_indices) == len(col_chunk_dims)

    # create a list of jobs

    jobs = __generate_jobs_dataframe(
        index_params=[channel_list, radii_list, row_chunk_indices, col_chunk_indices],
        index_names=["channel", "radius", "chunk_origin_0", "chunk_origin_1"],
    )

    # jobs_index = pd.MultiIndex.from_product(
    #         [channel_list, radii_list, row_chunk_indices, col_chunk_indices],
    #         names=['channel', 'radius', 'chunk_origin_0', 'chunk_origin_1']
    # )
    # jobs = DataFrame(
    #     index=jobs_index,
    #     columns=['channel', 'radius', 'chunk_origin_0', 'chunk_origin_1', 'chunk_dim_0', 'chunk_dim_1',
    #              'img_name', 'label', 'npoints', 'patchsize', 'chunksize', 'img_shm_name',
    #              'img_pixel_dtype', 'img_shape_0', 'img_shape_1', 'img_shape_2',
    #              'output_shm_name', 'output_offset', 'tmp_fpath',
    #              'patch_mask_shm_name', 'tmp_fpath_pixel']
    # )

    jobs_idx = pd.IndexSlice

    jobs["img_name"] = img_name

    channel_output_offset = 0
    for c in channel_list:
        jobs.loc[jobs_idx[c, :, :, :], "channel"] = c

        for chunk_i in row_chunk_indices:
            for chunk_j in col_chunk_indices:
                jobs.loc[jobs_idx[c, :, chunk_i, chunk_j], "output_offset"] = channel_output_offset + np.hstack(
                    [[0], nfeatures_cumsum[:-1]]
                )
        channel_output_offset += nfeatures_per_channel

    for idx_rr, rr in enumerate(radii_list):
        jobs.loc[jobs_idx[:, rr, :, :], "radius"] = rr
        jobs.loc[jobs_idx[:, rr, :, :], "npoints"] = npoints_list[idx_rr]

    for row_chunk_idx in row_chunk_indices:
        jobs.loc[jobs_idx[:, :, row_chunk_idx, :], "chunk_origin_0"] = row_chunk_idx
    jobs["chunk_origin_0"] = jobs["chunk_origin_0"].astype("uint32")  # TODO: DEBUG: idk why this became necessary
    for col_chunk_idx in col_chunk_indices:
        jobs.loc[jobs_idx[:, :, :, col_chunk_idx], "chunk_origin_1"] = col_chunk_idx
    jobs["chunk_origin_1"] = jobs["chunk_origin_1"].astype("uint32")

    # fill chunk dimensions column
    for idx_i, chunk_i in enumerate(row_chunk_indices):
        jobs.loc[jobs_idx[:, :, chunk_i, :], "chunk_dim_0"] = row_chunk_dims[idx_i]
    jobs["chunk_dim_0"] = jobs["chunk_dim_0"].astype("uint32")

    for idx_j, chunk_j in enumerate(col_chunk_indices):
        jobs.loc[jobs_idx[:, :, :, chunk_j], "chunk_dim_1"] = col_chunk_dims[idx_j]
    jobs["chunk_dim_1"] = jobs["chunk_dim_1"].astype("uint32")

    jobs["label"] = jobs.apply(
        lambda row: f"{img_name}_chunk_origin{row['chunk_origin_0']}-{row['chunk_origin_1']}_chunk_dim{row['chunk_dim_0']}-{row['chunk_dim_1']}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}",
        axis="columns",
    )

    img_name_pixel_cache = img_name + "_lbp_codes"
    jobs["pixel_cache_label"] = jobs.apply(
        lambda row: f"{img_name_pixel_cache}_chunk_origin{row['chunk_origin_0']}-{row['chunk_origin_1']}_chunk_dim{row['chunk_dim_0']}-{row['chunk_dim_1']}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}",
        axis="columns",
    )

    jobs["patchsize"] = patchsize
    jobs["chunksize"] = chunksize

    __fill_in_cache_dirs_in_jobs_df(
        jobs,
        pipeline_name,
        [histograms_cache_dir, lbp_codes_cache_dir],
        ["label", "pixel_cache_label"],
        ["tmp_fpath", "tmp_fpath_pixel"],
    )

    total_nfeatures = nfeatures_per_channel * len(channel_list)
    patch_features_shape = (nprows, npcols, total_nfeatures)
    jobs["total_nfeatures"] = total_nfeatures

    # Prepare contiguous array.
    # Channels will go first. Then h and w.
    img_data = np.ascontiguousarray(np.moveaxis(img_data, (0, 1, 2), (1, 2, 0)))

    log.info(f"run_chunked_fastlbp({pipeline_hash}): creating shared memory")
    # create shared memory for input image
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)

    # copy image to shared memory
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    np.copyto(input_img_np, img_data, casting="no")

    # copy mask to shared memory if provided.

    patch_mask_shm_result = __create_patch_mask_shm(
        nprows, npcols, patchsize, mask_method, pipeline_hash, img_mask, img_patch_mask, caller="run_chunked_fastlbp"
    )

    patch_mask = None
    patch_mask_shm = None

    if patch_mask_shm_result is not None:
        patch_mask, patch_mask_shm = patch_mask_shm_result

    # create and initialize shared memory for output
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype(_features_dtype).itemsize)
    )
    patch_features = np.ndarray(patch_features_shape, _features_dtype, buffer=patch_features_shm.buf)
    patch_features.fill(0)
    log.info(f"run_chunked_fastlbp({pipeline_hash}): shared memory created")

    jobs["img_shm_name"] = input_img_shm.name
    # jobs['img_mask_shm_name'] = img_mask_shm.name if img_mask_shm is not None else ""
    jobs["patch_mask_shm_name"] = patch_mask_shm.name if patch_mask_shm is not None else ""
    jobs["img_pixel_dtype"] = input_img_np.dtype  # note: always uint8
    jobs["img_shape_0"] = input_img_np.shape[0]  # nchannels
    jobs["img_shape_1"] = input_img_np.shape[1]  # h
    jobs["img_shape_2"] = input_img_np.shape[2]  # w
    jobs["output_shm_name"] = patch_features_shm.name

    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    if jobs_csv_savefile is not None:
        __dump_jobs_df_to_csv(jobs, jobs_csv_savefile)

    log.info(f"run_chunked_fastlbp({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s")
    log.info(f"run_chunked_fastlbp({pipeline_hash}): jobs:")
    log.info(jobs.head())
    log.info(f"Jobs DataFrame shape: {jobs.shape}")

    assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f"run_chunked_fastlbp({pipeline_hash}): start computation")
    t0 = time.perf_counter()

    # TODO: https://stackoverflow.com/questions/62748654/python-3-8-shared-memory-resource-tracker-producing-unexpected-warnings-at-appli
    # fix leaked shared memory warnings

    with Pool(ncpus) as pool:
        jobs_results = pool.map(func=__chunked_worker_fastlbp, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f"run_chunked_fastlbp({pipeline_hash}): computation finished in {t_elapsed:.5g}s. Start saving")

    # save results
    lbp_result = None
    output_fpath = os.path.join(outdir, outfile_name)
    output_abspath = os.path.abspath(output_fpath)

    if savefile:
        np.save(output_fpath, patch_features)
    else:
        lbp_result = patch_features.copy()

    __unlink_and_close_shms(input_img_shm, patch_features_shm, patch_mask_shm)

    log.info(f"run_chunked_fastlbp({pipeline_hash}): shared memory unlinked. Goodbye")

    # reset logger to its original level
    log.setLevel(DEFAULT_LEVEL)

    if savefile:
        return FastlbpResult(output_abspath, patch_mask)
    else:
        return lbp_result


def run_patch_fastlbp(
    img_data: ArrayLike,
    patch_coordinates_list: list[tuple[int, int]],
    radii_list: ArrayLike,
    npoints_list: ArrayLike,
    patchsize: int,
    ncpus: int = 1,
    img_name: str = "img_patch_lbp",
    jobs_csv_savefile: str | None = None,
    return_raw_features: bool = False,
    verbosity: int = 1,
) -> np.ndarray:

    logger_level = int_verbosity_to_logger_level(verbosity)
    log.setLevel(logger_level)

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) in [2, 3]
    assert img_data.dtype == np.uint8

    if len(img_data.shape) == 2:
        img_data = img_data[:, :, None]

    t = time.perf_counter()

    log.info("run_patched_fastlbp: initial setup...")

    ncpus = __get_optimal_ncpus(ncpus)

    log.info(f"run_fastlbp: initial setup took {time.perf_counter()-t:.5g}s")
    log.info(f"run_fastlbp: creating a list of jobs...")

    t = time.perf_counter()

    h, w, nchannels = img_data.shape
    nfeatures_cumsum = np.cumsum(np.array(npoints_list) + 2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    patch_features_result = []  # patch feature vectors in the same order as they passed as inputs

    # create a list of jobs
    jobs = __generate_jobs_dataframe(
        index_params=[channel_list, radii_list, patch_coordinates_list],
        index_names=["channel", "radius", "patch_center_coords"],
    )

    # # create a list of jobs
    # jobs_index = pd.MultiIndex.from_product(
    #         [channel_list, radii_list, patch_coordinates_list],
    #         names=['channel', 'radius', 'patch_center_coords']
    # )
    # jobs = pd.DataFrame(
    #     index=jobs_index,
    #     columns=['channel', 'radius', 'patch_center_coords',
    #              'img_name', 'label', 'npoints', 'patchsize', 'img_shm_name',
    #              'img_pixel_dtype', 'img_shape_0', 'img_shape_1', 'img_shape_2',
    #              'output_shm_name', 'output_offset']
    # )

    jobs_idx = pd.IndexSlice

    # TODO: use a separate DataFrame for static parameters shared by all jobs
    jobs["img_name"] = img_name
    jobs["return_raw_features"] = return_raw_features

    channel_output_offset = 0
    for c in channel_list:
        jobs.loc[jobs_idx[c, :, :, :], "channel"] = c

        for patch_center_coord_0, patch_center_coord_1 in patch_coordinates_list:
            jobs.loc[jobs_idx[c, :, [(patch_center_coord_0, patch_center_coord_1)]], "output_offset"] = (
                channel_output_offset + np.hstack([[0], nfeatures_cumsum[:-1]])
            )

        channel_output_offset += nfeatures_per_channel

    for idx_rr, rr in enumerate(radii_list):
        jobs.loc[jobs_idx[:, rr, :, :], "radius"] = rr
        jobs.loc[jobs_idx[:, rr, :, :], "npoints"] = npoints_list[idx_rr]

    for patch_center_coord_0, patch_center_coord_1 in patch_coordinates_list:
        jobs.loc[jobs_idx[:, :, [(patch_center_coord_0, patch_center_coord_1)]], "patch_center_coords"] = (
            f"{patch_center_coord_0} {patch_center_coord_1}"
        )

        str_coords_to_tuples = jobs.loc[
            jobs_idx[:, :, [(patch_center_coord_0, patch_center_coord_1)]], "patch_center_coords"
        ].apply(lambda x: tuple(map(int, x.split(" "))))

        jobs.loc[jobs_idx[:, :, [(patch_center_coord_0, patch_center_coord_1)]], "patch_center_coords"] = (
            str_coords_to_tuples
        )

    jobs["label"] = jobs.apply(
        lambda row: f"{img_name}_patch_center_coords{'_'.join(list(map(str, row['patch_center_coords'])))}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}",
        axis="columns",
    )

    jobs["patchsize"] = patchsize

    total_nfeatures = nfeatures_per_channel * len(channel_list)
    jobs["total_nfeatures"] = total_nfeatures

    # Prepare contiguous array.
    # Channels will go first. Then h and w.
    img_data = np.ascontiguousarray(np.moveaxis(img_data, (0, 1, 2), (1, 2, 0)))

    log.info(f"run_chunked_fastlbp: creating shared memory")
    # create shared memory for input image
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)

    # copy image to shared memory
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    np.copyto(input_img_np, img_data, casting="no")

    # create and initialize shared memory for output

    patch_result_shared_memory_list = []

    for patch_center_coord_0, patch_center_coord_1 in patch_coordinates_list:
        patch_features_shm = shared_memory.SharedMemory(
            create=True, size=(int(total_nfeatures) * np.dtype(_features_dtype).itemsize)
        )

        patch_features = np.ndarray((int(total_nfeatures),), _features_dtype, buffer=patch_features_shm.buf)
        patch_features.fill(0)

        jobs.loc[jobs_idx[:, :, [(patch_center_coord_0, patch_center_coord_1)]], "output_shm_name"] = (
            patch_features_shm.name
        )

        patch_result_shared_memory_list.append(patch_features_shm)
        patch_features_result.append(patch_features)

        log.info(f"run_single_patch_fastlbp: shared memory created")

    jobs["img_shm_name"] = input_img_shm.name
    jobs["img_pixel_dtype"] = input_img_np.dtype  # note: always uint8
    jobs["img_shape_0"] = input_img_np.shape[0]  # nchannels
    jobs["img_shape_1"] = input_img_np.shape[1]  # h
    jobs["img_shape_2"] = input_img_np.shape[2]  # w

    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    if jobs_csv_savefile is not None:
        __dump_jobs_df_to_csv(jobs, jobs_csv_savefile)

    log.info(f"run_patch_fastlbp: creating a list of jobs took {time.perf_counter()-t:.5g}s")
    log.info(f"run_patch_fastlbp: jobs:")
    log.info(jobs.head())
    log.info(f"Jobs DataFrame shape: {jobs.shape}")

    assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f"run_patch_fastlbp: start computation")
    t0 = time.perf_counter()
    with Pool(ncpus) as pool:
        raw_lbp_codes = pool.map(func=__single_patch_fastlbp_worker, iterable=jobs.iterrows())
    t_elapsed = time.perf_counter() - t0
    log.info(f"run_patch_fastlbp(): computation finished in {t_elapsed:.5g}s. Start saving")

    # save results

    result = []
    for patch_feat_res in patch_features_result:
        result.append(patch_feat_res.copy())

    __unlink_and_close_shms(input_img_shm, *patch_result_shared_memory_list)

    log.info(f"run_patch_fastlbp: shared memory unlinked. Goodbye")

    # reset logger to its original level
    log.setLevel(DEFAULT_LEVEL)

    if not return_raw_features:
        return result, None

    return result, raw_lbp_codes


def run_cuda_fastlbp(
    img_data: ArrayLike,
    radii_list: ArrayLike,
    npoints_list: ArrayLike,
    patchsize: int,
    chunksize: int | None = None,
    img_mask=None,
    img_patch_mask=None,
    mask_method="any",
    max_ram=None,
    img_name="img_chunked",
    savefile: str = "",
    overwrite_output: bool = False,
    jobs_csv_savefile: str | None = None,
    histograms_cache_dir: str | None = None,
    lbp_codes_cache_dir: str | None = None,
    return_raw_features: bool = False,
    verbosity: int = 1,
) -> Union[FastlbpResult, np.array]:

    import time
    from multiprocessing import Pool, shared_memory

    import pandas as pd
    from pandas import DataFrame

    from .common import _features_dtype, _raw_features_dtype
    from .utils import int_verbosity_to_logger_level, patchify_image_mask
    from .workers import __cuda_worker_fastlbp

    logger_level = int_verbosity_to_logger_level(verbosity)
    log.setLevel(logger_level)

    # validate params and prepare a pipeline
    assert len(radii_list) == len(npoints_list)
    assert len(img_data.shape) in [2, 3]
    assert img_data.dtype == np.uint8

    if len(img_data.shape) == 2:
        img_data = img_data[:, :, None]

    if img_mask is not None:
        assert img_mask.shape == img_data.shape[:2]
        assert img_mask.dtype == np.uint8

    t = time.perf_counter()

    log.info("run_cuda_fastlbp: initial setup...")

    img_name = __sanitize_img_name(img_name)
    outdir, outfile_name = os.path.dirname(savefile), os.path.basename(savefile)

    if outfile_name:
        outfile_name = __sanitize_outfile_name(outfile_name)

    # this way pipelines with different ncpus/radii/npoints can reuse tmp files if patchsize, img name and version are the same
    pipeline_hash = __create_pipeline_hash(
        "fastlbp-cuda",
        [
            str(img_data.shape),
            patchsize,
            chunksize,
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    # pixel cache consists of LBP codes before they are grouped into histograms
    # therefore it is invariant to the patch size parameter and
    # requires a separate hash that does not depend on patch size

    # UPDATE: the statement above is not correct for the chunked version
    # the chunks consist of a whole number of patches
    # meaning that changing the patch size will alter the splitting pattern
    # and the shape of the cached results
    # TODO: add separate pipeline hash for pixel cache in the regular function (not chunked)
    pipeline_hash_lbp_codes = __create_pipeline_hash(
        "fastlbp-cuda",
        [
            str(img_data.shape),
            patchsize,
            chunksize,
            "mask" if img_mask is not None else "no_mask",
            mask_method if img_mask is not None else "",
        ],
    )

    pipeline_name = f"{img_name}-fastlbp-cuda-{pipeline_hash}"
    pipeline_name_lbp_codes = f"{img_name}-fastlbp-cuda-{pipeline_hash_lbp_codes}"

    log.info("run_cuda_fastlbp: params:")
    log.info("img_shape, radii_list, npoints_list, patchsize, max_ram, img_name")
    log.info(f"{img_data.shape}, {radii_list}, {npoints_list}, {patchsize}, {chunksize}, {max_ram}, {img_name}")
    log.info(f"{histograms_cache_dir=}, {lbp_codes_cache_dir=}")
    if savefile:
        log.info(f"LBP is saved on disk in {savefile = }")
    log.info(f"pipeline hash is {pipeline_hash}")
    log.info(f"pipeline LBP codes hash is {pipeline_hash_lbp_codes}")

    if max_ram is not None:
        log.warning("max_ram parameter is currently ignored!")

    if savefile:
        res_outdir = outdir if outdir else os.getcwd()
        output_fpath = os.path.join(res_outdir, outfile_name)
        output_abspath = os.path.abspath(output_fpath)
        try:
            if os.path.exists(output_fpath) and not overwrite_output:
                log.error(
                    f"run_cuda_fastlbp({pipeline_hash}): overwrite_output is False and output file {output_abspath} already exists. Aborting."
                )
                return FastlbpResult(output_abspath, None)

            os.makedirs(res_outdir, exist_ok=True)
            if not os.access(res_outdir, os.W_OK):
                log.error(
                    f"run_cuda_fastlbp({pipeline_hash}): output dir {os.path.dirname(output_abspath)} is not writable. Aborting."
                )
                return FastlbpResult(output_abspath, None)
        except:
            log.error(
                f"run_cuda_fastlbp({pipeline_hash}): error accessing output dir {os.path.dirname(output_abspath)}. Aborting."
            )
            return FastlbpResult(output_abspath, None)

    log.info(f"run_cuda_fastlbp({pipeline_hash}): initial setup took {time.perf_counter()-t:.5g}s")
    log.info(f"run_cuda_fastlbp({pipeline_hash}): creating a list of jobs...")

    t = time.perf_counter()

    # method-specific params
    h, w, nchannels = img_data.shape
    nprows, npcols = h // patchsize, w // patchsize
    nfeatures_cumsum = np.cumsum(np.array(npoints_list) + 2)
    nfeatures_per_channel = nfeatures_cumsum[-1]
    channel_list = range(nchannels)

    # TODO: make more flexible
    nraw_features_cumsum = np.arange(len(npoints_list))  # used only if return_raw_features == True
    nraw_features_per_channel = len(npoints_list)  # used only if return_raw_features == True

    if chunksize is None:
        chunksize = max(nprows, npcols)

    # get chunk origins and shapes
    n_chunk_rows = nprows // chunksize
    n_chunk_cols = npcols // chunksize
    row_chunk_indices, col_chunk_indices = np.arange(n_chunk_rows), np.arange(n_chunk_cols)

    row_chunk_dims = np.full(n_chunk_rows, patchsize * chunksize)
    col_chunk_dims = np.full(n_chunk_cols, patchsize * chunksize)

    remaining_pixels_rows = (nprows % chunksize) * patchsize
    if remaining_pixels_rows > 0:
        row_chunk_indices = np.append(row_chunk_indices, [n_chunk_rows])
        row_chunk_dims = np.append(row_chunk_dims, [remaining_pixels_rows])

    remaining_pixels_cols = (npcols % chunksize) * patchsize
    if remaining_pixels_cols > 0:
        col_chunk_indices = np.append(col_chunk_indices, [n_chunk_cols])
        col_chunk_dims = np.append(col_chunk_dims, [remaining_pixels_cols])

    assert len(row_chunk_indices) == len(row_chunk_dims)
    assert len(col_chunk_indices) == len(col_chunk_dims)

    # create a list of jobs
    jobs_index = pd.MultiIndex.from_product(
        [channel_list, radii_list, row_chunk_indices, col_chunk_indices],
        names=["channel", "radius", "chunk_origin_0", "chunk_origin_1"],
    )
    jobs = DataFrame(
        index=jobs_index,
        columns=[
            "channel",
            "radius",
            "chunk_origin_0",
            "chunk_origin_1",
            "chunk_dim_0",
            "chunk_dim_1",
            "img_name",
            "label",
            "npoints",
            "patchsize",
            "chunksize",
            "img_shm_name",
            "img_pixel_dtype",
            "img_shape_0",
            "img_shape_1",
            "img_shape_2",
            "output_shm_name",
            "raw_output_shm_name",
            "output_offset",
            "raw_output_dimension",
            "tmp_fpath",
            "patch_mask_shm_name",
            "tmp_fpath_pixel",
        ],
    )

    jobs_idx = pd.IndexSlice

    jobs["img_name"] = img_name

    channel_output_offset = 0
    raw_features_channel_output_offset = 0
    # print(f'{jobs = } {channel_output_offset = } {nfeatures_per_channel = } DEBUG \n {107 + np.hstack([[0],nfeatures_cumsum[:-1]]) = }')
    # print(f'{row_chunk_indices = } {col_chunk_indices = } DBEUG')
    for c in channel_list:
        jobs.loc[jobs_idx[c, :, :, :], "channel"] = c

        for chunk_i in row_chunk_indices:
            for chunk_j in col_chunk_indices:
                jobs.loc[jobs_idx[c, :, chunk_i, chunk_j], "output_offset"] = channel_output_offset + np.hstack(
                    [[0], nfeatures_cumsum[:-1]]
                )
                jobs.loc[jobs_idx[c, :, chunk_i, chunk_j], "raw_output_dimension"] = (
                    raw_features_channel_output_offset + nraw_features_cumsum
                )
        channel_output_offset += nfeatures_per_channel
        raw_features_channel_output_offset += nraw_features_per_channel  # 1 layer per each channel

    for idx_rr, rr in enumerate(radii_list):
        jobs.loc[jobs_idx[:, rr, :, :], "radius"] = rr
        jobs.loc[jobs_idx[:, rr, :, :], "npoints"] = npoints_list[idx_rr]

    for row_chunk_idx in row_chunk_indices:
        jobs.loc[jobs_idx[:, :, row_chunk_idx, :], "chunk_origin_0"] = row_chunk_idx
    for col_chunk_idx in col_chunk_indices:
        jobs.loc[jobs_idx[:, :, :, col_chunk_idx], "chunk_origin_1"] = col_chunk_idx

    # fill chunk dimensions column
    for idx_i, chunk_i in enumerate(row_chunk_indices):
        jobs.loc[jobs_idx[:, :, chunk_i, :], "chunk_dim_0"] = row_chunk_dims[idx_i]

    for idx_j, chunk_j in enumerate(col_chunk_indices):
        jobs.loc[jobs_idx[:, :, :, chunk_j], "chunk_dim_1"] = col_chunk_dims[idx_j]

    jobs["label"] = jobs.apply(
        lambda row: f"{img_name}_chunk_origin{row['chunk_origin_0']}-{row['chunk_origin_1']}_chunk_dim{row['chunk_dim_0']}-{row['chunk_dim_1']}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}",
        axis="columns",
    )

    img_name_pixel_cache = img_name + "_lbp_codes"
    jobs["pixel_cache_label"] = jobs.apply(
        lambda row: f"{img_name_pixel_cache}_chunk_origin{row['chunk_origin_0']}-{row['chunk_origin_1']}_chunk_dim{row['chunk_dim_0']}-{row['chunk_dim_1']}_c{row.name[0]}_r{row.name[1]}_p{row['npoints']}",
        axis="columns",
    )

    jobs["patchsize"] = patchsize
    jobs["chunksize"] = chunksize

    # get the tmp path to read cached results from (if None then default is 'data/tmp')
    # the cache will be saved/ovewritten only if the corresponding flags are set to True
    # `save_results_cache`, `save_lbp_codes_cache`,

    # if histograms_cache_dir is set, save cached lbp results
    if histograms_cache_dir is not None:
        base_tmp_path = __get_tmp_dir_explicit(pipeline_name, histograms_cache_dir)
        jobs["tmp_fpath"] = jobs.apply(lambda row: os.path.join(base_tmp_path, row["label"]) + ".npy", axis="columns")
    else:
        jobs["tmp_fpath"] = ""

    # if lbp codes cache dir is set save raw LBP results
    if lbp_codes_cache_dir is not None:
        base_tmp_path_lbp_codes = __get_tmp_dir_explicit(pipeline_name_lbp_codes, lbp_codes_cache_dir)
        jobs["tmp_fpath_pixel"] = jobs.apply(
            lambda row: os.path.join(base_tmp_path_lbp_codes, row["pixel_cache_label"]) + ".npy", axis="columns"
        )
    else:
        jobs["tmp_fpath_pixel"] = ""

    total_nfeatures = nfeatures_per_channel * len(channel_list)
    total_raw_nfeatures = len(npoints_list) * len(channel_list)
    chunk_features_shape = (n_chunk_rows, n_chunk_rows, total_nfeatures)
    patch_features_shape = (nprows, npcols, total_nfeatures)
    raw_features_shape = (h, w, total_raw_nfeatures)
    jobs["total_nfeatures"] = total_nfeatures
    jobs["total_raw_nfeatures"] = total_raw_nfeatures

    # Prepare contiguous array.
    # Channels will go first. Then h and w.
    img_data = np.ascontiguousarray(np.moveaxis(img_data, (0, 1, 2), (1, 2, 0)))

    log.info(f"run_cuda_fastlbp({pipeline_hash}): creating shared memory")
    # create shared memory for input image
    input_img_shm = shared_memory.SharedMemory(create=True, size=img_data.nbytes)

    # copy image to shared memory
    input_img_np = np.ndarray(img_data.shape, img_data.dtype, input_img_shm.buf)
    np.copyto(input_img_np, img_data, casting="no")

    # copy mask to shared memory if provided.

    patch_mask_shm = None  # per patch mask

    patch_mask_shape = (nprows, npcols)
    patch_mask = None

    if img_mask is not None:
        log.info(f"run_cuda_fastlbp({pipeline_hash}): using image mask.")
        patch_mask = patchify_image_mask(img_mask, patchsize, edit_img_mask=False, method=mask_method)
        assert patch_mask.shape == patch_mask_shape

        patch_mask_shm = shared_memory.SharedMemory(create=True, size=patch_mask.nbytes)
        patch_mask_np = np.ndarray(patch_mask_shape, dtype=np.uint8, buffer=patch_mask_shm.buf)
        np.copyto(patch_mask_np, patch_mask, casting="no")

        log.info(
            f"run_cuda_fastlbp({pipeline_hash}): pixel mask converted to patch mask. Created shared memory for patch mask."
        )
    elif img_patch_mask is not None:
        log.info(f"run_cuda_fastlbp({pipeline_hash}): using patch mask.")
        patch_mask = img_patch_mask

        assert patch_mask.shape == patch_mask_shape

        patch_mask_shm = shared_memory.SharedMemory(create=True, size=patch_mask.nbytes)
        patch_mask_np = np.ndarray(patch_mask_shape, dtype=np.uint8, buffer=patch_mask_shm.buf)
        np.copyto(patch_mask_np, patch_mask, casting="no")

        log.info(f"run_cuda_fastlbp({pipeline_hash}): created shared memory for patch mask.")

    # create and initialize shared memory for output
    patch_features_shm = shared_memory.SharedMemory(
        create=True, size=(int(np.prod(patch_features_shape)) * np.dtype(_features_dtype).itemsize)
    )

    patch_features = np.ndarray(patch_features_shape, _features_dtype, buffer=patch_features_shm.buf)
    patch_features.fill(0)

    raw_features_shm = None
    raw_features = None
    if return_raw_features:
        raw_features_shm = shared_memory.SharedMemory(
            create=True, size=(int(np.prod(raw_features_shape)) * np.dtype(_raw_features_dtype).itemsize)
        )
        raw_features = np.ndarray(raw_features_shape, _raw_features_dtype, buffer=raw_features_shm.buf)
        raw_features.fill(0)

    log.info(f"run_cuda_fastlbp({pipeline_hash}): shared memory created")

    jobs["img_shm_name"] = input_img_shm.name
    # jobs['img_mask_shm_name'] = img_mask_shm.name if img_mask_shm is not None else ""
    jobs["patch_mask_shm_name"] = patch_mask_shm.name if patch_mask_shm is not None else ""
    jobs["img_pixel_dtype"] = input_img_np.dtype  # note: always uint8
    jobs["img_shape_0"] = input_img_np.shape[0]  # nchannels
    jobs["img_shape_1"] = input_img_np.shape[1]  # h
    jobs["img_shape_2"] = input_img_np.shape[2]  # w
    jobs["output_shm_name"] = patch_features_shm.name
    jobs["raw_output_shm_name"] = None if raw_features_shm is None else raw_features_shm.name

    # Sort jobs starting from the longest ones, i.e. from larger radii to smaller ones.
    # `level=1` values are radii
    jobs.sort_index(level=1, ascending=False, inplace=True)

    if jobs_csv_savefile is not None:
        jobs_csv_savedir = os.path.dirname(jobs_csv_savefile)
        if jobs_csv_savedir:
            os.makedirs(jobs_csv_savedir, exist_ok=True)
        jobs.to_csv(jobs_csv_savefile)

    log.info(f"run_cuda_fastlbp({pipeline_hash}): creating a list of jobs took {time.perf_counter()-t:.5g}s")
    log.info(f"run_cuda_fastlbp({pipeline_hash}): jobs:")
    log.info(jobs.head())
    log.info(f"Jobs DataFrame shape: {jobs.shape}")

    # assert jobs.isna().sum().sum() == 0

    # compute

    log.info(f"run_cuda_fastlbp({pipeline_hash}): start computation")
    t0 = time.perf_counter()

    for rrow in jobs.iterrows():
        __cuda_worker_fastlbp(rrow)

    t_elapsed = time.perf_counter() - t0
    log.info(f"run_cuda_fastlbp({pipeline_hash}): computation finished in {t_elapsed:.5g}s. Start saving")

    # save results
    lbp_result = None

    raw_lbp_result = None
    if raw_features is not None:
        raw_lbp_result = raw_features.copy()  # DEBUG

    if savefile:
        np.save(output_fpath, patch_features)
    else:
        lbp_result = patch_features.copy()

    input_img_shm.unlink()
    input_img_shm.close()
    patch_features_shm.unlink()
    patch_features_shm.close()

    if raw_features_shm is not None:
        raw_features_shm.unlink()
        raw_features_shm.close()

    # if img_mask_shm is not None:
    #     img_mask_shm.unlink()
    if patch_mask_shm is not None:
        patch_mask_shm.unlink()
        patch_mask_shm.close()

    log.info(f"run_cuda_fastlbp({pipeline_hash}): shared memory unlinked. Goodbye")

    # reset logger to its original level
    log.setLevel(DEFAULT_LEVEL)

    if savefile:
        return FastlbpResult(output_abspath, patch_mask), raw_lbp_result
    else:
        return lbp_result, raw_lbp_result
