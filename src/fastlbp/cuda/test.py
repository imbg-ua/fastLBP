import time

import lbp_cuda
import numpy as np

t = time.perf_counter()
a = np.arange(1600_000_000, dtype=np.uint8).reshape((40_000, 40_000))
b = np.empty((40_000, 40_000), dtype=np.uint32)
lbp_cuda.cuda_lbp(a, b, 8, 1)
print(f"elapsed time {time.perf_counter() - t}")
# print(f'{b = }')
np.save(f"test_b.npy", b)
