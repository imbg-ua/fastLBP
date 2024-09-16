import numpy as np
import skimage as ski
from fastlbp_imbg import run_fastlbp, get_p_for_r
import unittest
from PIL import Image
import os

class TestFastLBP(unittest.TestCase):

    def test_small(self):
        Rs = [1,2,3,4,5]
        Ps = get_p_for_r(Rs)

        data = np.asarray(Image.open("tests/tex1-smol.tif"))
        result = run_fastlbp(
            data,
            Rs, Ps, patchsize=50,
            ncpus=1,
            img_name='TestFastLBP_small',
            outfile_name="TestFastLBP_small_features.npy",
            save_intermediate_results=False
        )
        self.assertIsNone(result.patch_mask)
        actual_features = np.load(result.output_abspath)
        expected_features = np.load("tests/expected_small_features.npy")
        self.assertTrue((actual_features == expected_features).all())
        os.remove(result.output_abspath)


if __name__ == '__main__':
    unittest.main()
