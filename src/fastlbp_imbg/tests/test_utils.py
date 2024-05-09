import numpy as np
from ..utils import (
    get_patch,
    patchify_image_mask,
    get_reduced_hist_masks,
    reduce_histogram,
)
import unittest
from math import floor

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

    def test_reduced_hist_masks(self):
        for P in range(7,200+1):
            flat, corner, edge, nonuniform = get_reduced_hist_masks(P)
            X = np.array([flat, corner, edge, nonuniform])
            with self.subTest(P=P):
                # flat,corner,edge,nonuniform should form a partition. That is, they must not intersect.
                self.assertFalse(any(flat & corner))
                self.assertFalse(any(flat & edge))
                self.assertFalse(any(flat & nonuniform))
                self.assertFalse(any(corner & edge))
                self.assertFalse(any(corner & nonuniform))
                self.assertFalse(any(edge & nonuniform))

                # check important values
                self.assertTrue(flat[0] == 1)
                self.assertTrue(flat[P] == 1)
                self.assertTrue(edge[floor(P/2)] == 1)
                self.assertTrue(nonuniform[-1] == 1)

                # check if partition is the same as in paper
                bins = np.floor(np.linspace(0,P,6))[1:]
                groups = np.digitize(np.arange(P+1+1), bins, True)
                groups[groups == 4] = 0
                groups[groups == 3] = 1
                groups += 1

                actual = (flat + 2*corner + 3*edge + 6*nonuniform)

                self.assertTrue(all(groups == actual), f"\n  true: {groups} \n utils: {actual}")

    def test_lbp_reduced_hist(self):
        # check that reduce_histogram function behaves correctly.
        # Function is implemented as a matrix multiplication.
        # I check whether the feature in the output that will be incremented is correct for each code p from 0 to P+1 for P from some random vals. 
        for P in [7, 14, 20, 27, 30]:
            bins = get_reduced_hist_masks(P)
            for p in range(P+1+1):
                hist = np.zeros(P+2)
                hist[p] = 1
                group_num = np.max([1*bins[0][p], 2*bins[1][p], 3*bins[2][p], 4*bins[3][p]]) - 1
                ans = reduce_histogram(hist, *bins)
                self.assertTrue(all(ans == np.eye(4)[group_num]), f"P = {P}, p = {p}, group_num = {group_num}, ans = {ans}")

if __name__ == '__main__':
    unittest.main()
