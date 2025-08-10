import numpy as np
import lbp_cuda
a = np.arange(9, dtype=np.uint8).reshape((3, 3))
b = np.empty((3, 3), dtype=np.uint32)
lbp_cuda.cuda_lbp(a, b, 3, 1)
print(f'{b = }')
