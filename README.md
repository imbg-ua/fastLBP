# fastLBP
Highly parallel LBP implementation

## Installation

Підготовка середовища пітона (краще робити **не** на head node)
- Знайти десь Python 3.11 (напр. `conda create -n p11 python=3.11`)
- Перевірити що середовище правильне 
	- `python --version` та `pip --version`
- Встановити пакет через 
	- `pip install fastlbp-imbg`
- Або вручну через 
	- `git clone git@github.com:imbg-ua/fastLBP.git`
	- `pip install fastLBP`

Наша директорія на lustre
`/lustre/scratch126/casm/team268im/`



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
- Use `max_ram` parameter to estimate optimal number of sub-processes and collect memory stats. Now `max_ram` is ignored.

## Planned modules
### run_chunked_skimage
Similar to [1. run_skimage](#1-run_skimage), but each subprocess should compute LBP for its image chunk, not the whole image.

### run_dask and run_chunked_dask
Similar to [1. run_skimage](#1-run_skimage), but use [Dask](https://docs.dask.org/en/stable/) and [`dask.array.map_overlap`](https://docs.dask.org/en/stable/generated/dask.array.map_overlap.html#dask.array.map_overlap) for parallelisation instead of `multiprocessing` and manual data wrangling

## Other notable things to try
- Perform benchmarking of an in-house optimised cython version of `skimage.feature.local_binary_pattern` (see [skimage_lbp.pyx at imbg-ua/nf-img-benchmarking](https://github.com/imbg-ua/nf-img-benchmarking/blob/main/skimage_lbp/skimage_lbp.pyx))
- Do some research on Numba - is it applicable here?
- Add annotations to run_skimage results using [`anndata`](https://anndata.readthedocs.io/en/latest/)
