import numpy as np
import skimage as ski
from ..lbp import uniform_lbp_uint8
from ..fastlbp import get_p_for_r
import unittest

class TestCythonLBP(unittest.TestCase):

    def test_small(self):
        Rs = np.arange(2, 15)
        Ps = [get_p_for_r(r) for r in Rs]

        for i in range(5):
            H,W = np.random.randint(100,500,2)
            data = (np.random.rand(H,W)*256).astype(np.uint8)

            for r,p in zip(Rs, Rs):
                skimage_result = ski.feature.local_binary_pattern(data, P=p, R=r, method='uniform').astype(np.uint16)
                cython_result = uniform_lbp_uint8(data, P=p, R=r)
                with self.subTest(i=i, radius=r, npoints=p, height=H, weight=W):
                    self.assertEqual(cython_result.dtype, skimage_result.dtype, "Dtype mismatch")
                    self.assertTrue((cython_result == skimage_result).all)

    def test_big(self):
        Rs = np.arange(2, 5)
        Ps = [get_p_for_r(r) for r in Rs]

        for i in range(5):
            H,W = np.random.randint(1000,10_000,2)
            data = (np.random.rand(H,W)*256).astype(np.uint8)

            for r,p in zip(Rs, Rs):
                skimage_result = ski.feature.local_binary_pattern(data, P=p, R=r, method='uniform').astype(np.uint16)
                cython_result = uniform_lbp_uint8(data, P=p, R=r)
                with self.subTest(i=i, radius=r, npoints=p, height=H, weight=W):
                    self.assertEqual(cython_result.dtype, skimage_result.dtype, "Dtype mismatch")
                    self.assertTrue((cython_result == skimage_result).all)



if __name__ == '__main__':
    unittest.main()
