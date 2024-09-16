# fastLBP
Highly parallel LBP implementation

> **Important pre-release warning**:
> If aborted mid-execution, this software sometimes create a lot of orphan processes that needs to be killed manually.
> Please, note down the name of your python script, search for `Python` in your task manager and look for the processes that correspond to your python script.

## Requirements
FastLBP is tested with Python 3.11 on Windows 10, Debian 11, and Ubuntu 22.04

Python requirements are:
- numpy >= 1.26.0
- Cython (to build the binary modules, will be optional in the future)
- scikit-image >= 0.22.0 (mostly for testing, we plan making this requirement optional in the future)
- pandas >= 2.1.1
- psutil

## Installation

- Activate or create a Python 3.11 environment (e.g. using `conda create -y -n p11 python=3.11 && conda activate p11`)
- Verify you are using the right env
	- `python --version` and `pip --version`
- Install a stable version from PyPI  
	`pip install fastlbp-imbg`
- Or build the latest version from sources  
	```
	git clone git@github.com:imbg-ua/fastLBP.git
	cd fastLBP
	# git checkout <branchname> # if you need a specific branch
	pip install . # this will install the fastlbp_imbg package in the current env
	```
- You can use `import fastlbp_imbg as fastlbp` now

## Testing
```
# in repo root
conda activate fastlbp
pip install -e .
python -m unittest
```

## Bug reporting
You can report a bug or suggest an improvement using [our github issues](https://github.com/imbg-ua/fastLBP/issues)

## Implemented modules
### run_fastlbp
Computes multiradial LBP of a single multichannel image in a parallel fashion.

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
Similar to [1. run_fastlbp](#1-run_fastlbp), but each subprocess should compute LBP for its image chunk, not the whole image.

### run_dask and run_chunked_dask
Similar to [1. run_fastlbp](#1-run_fastlbp), but use [Dask](https://docs.dask.org/en/stable/) and [`dask.array.map_overlap`](https://docs.dask.org/en/stable/generated/dask.array.map_overlap.html#dask.array.map_overlap) for parallelisation instead of `multiprocessing` and manual data wrangling
