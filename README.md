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

## Testing
```
# in repo root
conda activate fastlbp
pip install -e .
python -m unittest
```

## Implemented modules
### run_fastlbp
Computes multiradial LBP of a single multichannel image in a parallel fashion.  
This is a quick sample implementation that could be a baseline for further benchmarking.

Features:
- Powered by `fastlbp_imbg.lbp`, our implementation of `skimage.feature.local_binary_pattern`
- Concurrency is managed by Python's [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) module
- Parallel computation via `multiprocessing.Pool` of size `ncpus`
- Efficient memory usage via  `multiprocessing.shared_memory` to make sure processes do not create redundant copies of data
- If `save_intermediate_results=False` then computes everything in RAM, no filesystem usage

TODO: 
- Use `max_ram` parameter to estimate optimal number of sub-processes and collect memory stats. Now `max_ram` **is ignored**.

## Planned modules
### run_chunked_skimage
Similar to [1. run_skimage](#1-run_skimage), but each subprocess should compute LBP for its image chunk, not the whole image.

### run_dask and run_chunked_dask
Similar to [1. run_skimage](#1-run_skimage), but use [Dask](https://docs.dask.org/en/stable/) and [`dask.array.map_overlap`](https://docs.dask.org/en/stable/generated/dask.array.map_overlap.html#dask.array.map_overlap) for parallelisation instead of `multiprocessing` and manual data wrangling
