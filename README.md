# fastLBP
Highly parallel LBP implementation

## Installation

Note: it is not recommended to proceed on a head node; consider starting an ijob or a jupyter instance.

- Activate a Python 3.11 environment (e.g. using `conda create -y -n p11 python=3.11 && conda activate p11`)
- Verify you are using the right env
	- `python --version` та `pip --version`
- Install a stable version from PyPI
	- `pip install fastlbp-imbg`
- Or build the latest version from sources
```
git clone git@github.com:imbg-ua/fastLBP.git
cd fastLBP
# git checkout <branchname> # if you need a specific branch
pip install . # this will install the fastlbp_imbg package in the current env
```
- You can use `import fastlbp_imbg` now

#### Misc
Our lustre directory is
`/lustre/scratch126/casm/team268im/`


## Cython2 branch details
TODO:
- [x] Data Contiguity (`e3293cd`)
- [x] Job sorting (from slow to quick) in fastlbp.py (`8a21a04`)
- [ ] Change data types in lbp.pyx from float64 to something smaller (`IN PROGRESS`)
- [ ] Add mask feature to lbp.pyx
- [ ] Add mask feature to fastlbp.py
- [ ] Fill all commit numbers


## Implemented modules
### run_skimage
Computes multiradial LBP of a single multichannel image in a parallel fashion.  
This is a quick sample implementation that could be a baseline for further benchmarking.

Features:
- Powered by `skimage.feature.local_binary_pattern`
- Concurrency is managed by Python's [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) module
- Parallel computation via `multiprocessing.Pool` of size `ncpus`
- Efficient memory usage via  `multiprocessing.shared_memory` to make sure processes do not create redundant copies of data
- It computes everything in RAM, no filesystem usage

TODO: 
- Use `max_ram` parameter to estimate optimal number of sub-processes and collect memory stats. Now `max_ram` **is ignored**.

## Planned modules
### run_chunked_skimage
Similar to [1. run_skimage](#1-run_skimage), but each subprocess should compute LBP for its image chunk, not the whole image.

### run_dask and run_chunked_dask
Similar to [1. run_skimage](#1-run_skimage), but use [Dask](https://docs.dask.org/en/stable/) and [`dask.array.map_overlap`](https://docs.dask.org/en/stable/generated/dask.array.map_overlap.html#dask.array.map_overlap) for parallelisation instead of `multiprocessing` and manual data wrangling

## Other notable things to try
- Perform benchmarking of an in-house optimised cython version of `skimage.feature.local_binary_pattern` (see [skimage_lbp.pyx at imbg-ua/nf-img-benchmarking](https://github.com/imbg-ua/nf-img-benchmarking/blob/main/skimage_lbp/skimage_lbp.pyx))
- Do some research on Numba - is it applicable here?
- Add annotations to run_skimage results using [`anndata`](https://anndata.readthedocs.io/en/latest/)
