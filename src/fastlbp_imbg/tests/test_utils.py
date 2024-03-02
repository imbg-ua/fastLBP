import numpy as np
from ..utils import (
    get_patch,
    patchify_image_mask,
)
import unittest

class TestFastlbpUtils(unittest.TestCase):
    def test_get_patch(self):
        data = np.arange(2701).reshape(37,73)
        patchsize = 5
        r,c = 4, 5
        self.assertTrue(
            (get_patch(data, patchsize, r, c) == data[(r*patchsize):((r+1)*patchsize), (c*patchsize):((c+1)*patchsize)]).all(),
            "invalid read"
        )

        data = np.zeros((9,9), dtype=np.uint8)
        patchsize = 3
        patch = get_patch(data, patchsize, 1, 1)
        patch.fill(1)
        expected = np.zeros((9,9), dtype=np.uint8)
        expected[3:6,3:6] = 1
        self.assertTrue((data == expected).all(), "invalid write")

    def test_complete_background_mask(self):
        mask = np.array([
            [0, 0,  1, 0,  1, 1,  5],
            [0, 0,  0, 0,  0, 1,  6],

            [1, 1,  2, 0,  0, 2,  7],
            [1, 1,  0, 0,  0, 0,  8],

            [0, 1,  1, 2,  0, 0,  9],
            [2, 3,  3, 4,  0, 0,  10],

            [11, 12, 13, 14, 15, 16, 17]
        ])
        patchsize = 2

        expected_include_patchmask = np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ])
        expected_include_mask = np.array([
            [0, 0,  1, 1,  1, 1,  5],
            [0, 0,  1, 1,  1, 1,  6],

            [1, 1,  1, 1,  1, 1,  7],
            [1, 1,  1, 1,  1, 1,  8],

            [1, 1,  1, 2,  0, 0,  9],
            [1, 1,  3, 4,  0, 0,  10],

            [11, 12, 13, 14, 15, 16, 17]
        ])

        expected_exclude_patchmask = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        expected_exclude_mask = np.array([
            [0, 0,  0, 0,  0, 0,  5],
            [0, 0,  0, 0,  0, 0,  6],

            [1, 1,  0, 0,  0, 0,  7],
            [1, 1,  0, 0,  0, 0,  8],

            [0, 0,  1, 2,  0, 0,  9],
            [0, 0,  3, 4,  0, 0,  10],

            [11, 12, 13, 14, 15, 16, 17]
        ])

        A = mask.copy()
        patchmask_include = patchify_image_mask(A, patchsize=2, edit_img_mask=True, method='any')
        self.assertTrue((A == expected_include_mask).all())
        self.assertTrue((patchmask_include == expected_include_patchmask).all())
        
        B = mask.copy()
        patchmask_exclude = patchify_image_mask(B, patchsize=2, edit_img_mask=True, method='all')
        self.assertTrue((B == expected_exclude_mask).all())
        self.assertTrue((patchmask_exclude == expected_exclude_patchmask).all())


if __name__ == '__main__':
    unittest.main()
