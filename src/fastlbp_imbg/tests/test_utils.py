import numpy as np
from ..utils import (
    get_patch,
    patchify_image_mask,
    get_reduced_hist_masks,
    get_reduction_matrix,
    reduce_features,
    hist_masks_as_tuple,
    ReducedHistMasks,
    MinimalHistMasks,
)
from ..fastlbp import get_radii, get_p_for_r
import unittest
import itertools
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

    def test_minimal_histogram_masks(self):
        with self.subTest('no reduction for small P', P=7):
            P = 7
            histogram = np.arange(P+2)

            rhm_m = get_reduced_hist_masks(P, method='minimal')
            self.assertTrue(all(rhm_m == np.eye(P+2)))
            self.assertTrue(all(reduce_features(rhm_m, histogram) == histogram))

            rhm_r = get_reduced_hist_masks(P, method='reduced')
            self.assertTrue(all(rhm_r == np.eye(P+2)))
            self.assertTrue(all(reduce_features(rhm_r, histogram) == histogram))

        for P in [15, 25, 49, 101]:
            with self.subTest(P=P, method='minimal'):
                generated_minimal_masks = get_reduced_hist_masks(P, method='minimal')
            
                self.assertTrue(generated_minimal_masks.shape == (4, P+2))

                # flat,corner,edge,nonuniform should form a partition. That is, they must not intersect.
                for row1, row2 in itertools.combinations(generated_minimal_masks):
                    self.assertFalse(any(row1 & row2), 'masks are not orthogonal')

                hmt = hist_masks_as_tuple(generated_minimal_masks)
                self.assertTrue(isinstance(hmt, MinimalHistMasks), "wrong type")

                # check important values
                self.assertTrue(hmt.flat[0] == 1)
                self.assertTrue(hmt.flat[P] == 1)
                self.assertTrue(hmt.edge[floor(P/2)] == 1)
                self.assertTrue(hmt.nonuniform[-1] == 1)

                # check if partition is the same as in paper
                bins = np.floor(np.linspace(0,P,6))[1:]
                groups = np.digitize(np.arange(P+2), bins, True)
                groups[groups == 4] = 0
                groups[groups == 3] = 1
                groups += 1
                test = (hmt.flat + 2*hmt.corner + 3*hmt.edge + 6*hmt.nonuniform)

                self.assertTrue(all(groups == test), f"\n  true: {groups} \n test: {test}")
                
            with self.subTest(P=P, method='reduced'):
                generated_minimal_masks = get_reduced_hist_masks(P, method='reduced')
            
                self.assertTrue(generated_minimal_masks.shape == (6, P+2))

                # flat,corner,edge,nonuniform should form a partition. That is, they must not intersect.
                for row1, row2 in itertools.combinations(generated_minimal_masks):
                    self.assertFalse(any(row1 & row2), 'masks are not orthogonal')

                hmt = hist_masks_as_tuple(generated_minimal_masks)
                self.assertTrue(isinstance(generated_minimal_masks, ReducedHistMasks), "wrong type")

                # check important values
                self.assertTrue(hmt.flat_lo[0] == 1)
                self.assertTrue(hmt.flat_hi[P] == 1)
                self.assertTrue(hmt.edge[floor(P/2)] == 1)
                self.assertTrue(hmt.nonuniform[-1] == 1)

                # check if partition is the same as in paper
                bins = np.floor(np.linspace(0,P,6))[1:]
                groups = np.digitize(np.arange(P+2), bins, True)
                groups += 1
                test = (hmt.flat_lo + 2*hmt.corner_lo + 3*hmt.edge + 4*hmt.corner_hi + 5*hmt.flat_hi + 6*hmt.nonuniform)

                self.assertTrue(all(groups == test), f"\n  true: {groups} \n test: {test}")

    def test_histogram_reduction(self):
        # Test feature reduction for a single histogram. 
        # That is, a single patch and a single LBP run.

        # No reduction for P<10
        with self.subTest(P=7, method='minimal'):
                P = 7
                hm = get_reduced_hist_masks(P, method='minimal')
                hmt = hist_masks_as_tuple(hm)
                hist = np.arange(P+2)

                test_reduced_hist = reduce_features(hist, hm)
                self.assertTrue(test_reduced_hist.shape == (P+2,))
                self.assertTrue(all(test_reduced_hist == hist), f"\n true: {true_reduced_hist} \n test: {test_reduced_hist}")
        with self.subTest(P=7, method='reduced'):
                P = 7
                hm = get_reduced_hist_masks(P, method='reduced')
                hmt = hist_masks_as_tuple(hm)
                hist = np.arange(P+2)

                test_reduced_hist = reduce_features(hist, hm)
                self.assertTrue(test_reduced_hist.shape == (P+2,))
                self.assertTrue(all(test_reduced_hist == hist), f"\n true: {true_reduced_hist} \n test: {test_reduced_hist}")

        for P in [14, 20, 27, 30]:
            with self.subTest(P=P, method='minimal'):
                hm = get_reduced_hist_masks(P, method='minimal')
                hmt = hist_masks_as_tuple(hm)
                hist = np.arange(P+2)
                true_reduced_hist = np.zeros(4)

                for mask_i, mask in enumerate(hmt):
                    true_reduced_hist[mask_i] = hist[mask].sum()

                test_reduced_hist = reduce_features(hist, hm)
                self.assertTrue(test_reduced_hist.shape == (4,))
                self.assertTrue(all(test_reduced_hist == true_reduced_hist), f"\n true: {true_reduced_hist} \n test: {test_reduced_hist}")
                
            with self.subTest(P=P, method='reduced'):
                hm = get_reduced_hist_masks(P)
                hmt = hist_masks_as_tuple(hm, method='reduced')
                hist = np.arange(P+2)
                true_reduced_hist = np.zeros(6)

                for mask_i, mask in enumerate(hmt):
                    true_reduced_hist[mask_i] = hist[mask].sum()

                test_reduced_hist = reduce_features(hist, hm)
                self.assertTrue(test_reduced_hist.shape == (6,))
                self.assertTrue(all(test_reduced_hist == true_reduced_hist), f"\n true: {true_reduced_hist} \n test: {test_reduced_hist}")
                
    def test_full_features_reduction(self):
        # Test feature reduction for a whole feature array (n_patch_rows, n_patch_cols, n_features). 
        # That is, all patches and all LBP runs.

        def true_minimal_reduction(npoints, features):
            generated_minimal_masks = get_reduced_hist_masks(P, method='minimal')
            nfeatures = np.array(npoints) + 2
            nfeatures[nfeatures >= (10+2)] = 4
            reduced_features_num = nfeatures.sum()
            nrows,ncols,nfeatures = features.shape
            reduced_features = np.zeros((nrows, ncols, reduced_features_num))
            for r,c in itertools.product(range(nrows), range(ncols)):
                for mask_i,mask in enumerate(generated_minimal_masks):
                    reduced_features[r,c,mask_i] = features[r,c,mask].sum()
            return reduced_features
        
        def true_reduction(npoints, features):
            generated_reduced_masks = get_reduced_hist_masks(P, method='reduced')
            nfeatures = np.array(npoints) + 2
            nfeatures[nfeatures >= (10+2)] = 6
            reduced_features_num = nfeatures.sum()
            nrows,ncols,nfeatures = features.shape
            reduced_features = np.zeros((nrows, ncols, reduced_features_num))
            for r,c in itertools.product(range(nrows), range(ncols)):
                for mask_i,mask in enumerate(generated_reduced_masks):
                    reduced_features[r,c,mask_i] = features[r,c,mask].sum()
            return reduced_features

        ncols, nrows = 3,7        
        npoints = np.arange(25)
        nfeatures = (npoints + 2).sum()
        features = np.random.randint(0,100,size=(ncols,nrows,nfeatures))

        with self.subTest(method='minimal'):
            reduction_matrix = get_reduction_matrix(npoints, method='minimal')
            test_reduced_features = reduce_features(features, reduction_matrix)
            true_reduced_features = true_minimal_reduction(npoints, features)
            self.assertTrue(test_reduced_features.shape == true_reduced_features.shape)
            self.assertTrue(all(test_reduced_features == true_reduced_features))
            
        with self.subTest(method='reduced'):
            reduction_matrix = get_reduction_matrix(npoints, method='reduced')
            test_reduced_features = reduce_features(features, reduction_matrix)
            true_reduced_features = true_reduction(npoints, features)
            self.assertTrue(test_reduced_features.shape == true_reduced_features.shape)
            self.assertTrue(all(test_reduced_features == true_reduced_features))



if __name__ == '__main__':
    unittest.main()
