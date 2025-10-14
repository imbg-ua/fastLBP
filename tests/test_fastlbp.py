import numpy as np
from fastlbp import run_fastlbp, get_p_for_r
import unittest
from PIL import Image
import os
import shutil

class TestFastLBP(unittest.TestCase):

    def setUp(self):
        self.datadir = os.path.abspath("./data")
        if os.path.isdir(self.datadir):
            shutil.rmtree(self.datadir)

    def test_small(self):
        Rs = [1,2,3,4,5]
        Ps = get_p_for_r(Rs)

        expected_output_abspath = os.path.abspath("./data/out/TestFastLBP_small_features.npy")
        self.assertFalse(os.path.exists(expected_output_abspath), f"Output file {expected_output_abspath} already exists. Initial cleanup somehow failed.")

        data = np.asarray(Image.open("tests/tex1-smol.tif"))
        result = run_fastlbp(
            data,
            Rs, Ps, patchsize=50,
            ncpus=1,
            img_name='TestFastLBP_small',
            outfile_name="TestFastLBP_small_features.npy",
            save_intermediate_results=False
        )

        self.assertEqual(result.output_abspath, expected_output_abspath, "Invalid output file path")
        self.assertIsNone(result.patch_mask)
        actual_features = np.load(result.output_abspath)
        expected_features = np.load("tests/expected_small_features.npy")
        self.assertTrue((actual_features == expected_features).all())
        

    def test_basic_cache_idempotency(self):
        Rs = [1,2,3,4,5]
        Ps = get_p_for_r(Rs)

        tmpabspath = os.path.abspath("./data/tmp")
        self.assertFalse(os.path.isdir("./data/tmp"), f"Tmp directory {tmpabspath} already exists. Initial cleanup somehow failed.")

        # First run
        data = np.asarray(Image.open("tests/tex1-smol.tif"))
        result = run_fastlbp(
            data,
            Rs, Ps, patchsize=50,
            ncpus=1,
            img_name='TestFastLBP_small',
            outfile_name="TestFastLBP_small_features.npy",
            save_intermediate_results=True # create cache,
        )
        self.assertIsNone(result.patch_mask)
        actual_features = np.load(result.output_abspath)
        expected_features = np.load("tests/expected_small_features.npy")
        self.assertTrue((actual_features == expected_features).all())

        self.assertTrue(os.path.isdir("./data/tmp"), f"tmp directory {tmpabspath} does not exist.")

        # Second run with the same parameters. Should give the same result.
        # Replace original image with zeros array of same shape to ensure run_fastlbp is using cache and not actual image data.
        # This will not affect hash, as it is only affected by image size, not its contents.

        os.remove(result.output_abspath)

        data = np.asarray(Image.open("tests/tex1-smol.tif"))
        fake_data = np.zeros_like(data)
        result = run_fastlbp(
            fake_data,
            Rs, Ps, patchsize=50,
            ncpus=1,
            img_name='TestFastLBP_small',
            outfile_name="TestFastLBP_small_features.npy",
            save_intermediate_results=True # use cache
        )
        self.assertIsNone(result.patch_mask)
        actual_features = np.load(result.output_abspath)
        expected_features = np.load("tests/expected_small_features.npy")
        self.assertTrue((actual_features == expected_features).all(), "Cache malfunction!")

    def tearDown(self):
        if os.path.isdir(self.datadir):
            shutil.rmtree(self.datadir)


if __name__ == '__main__':
    unittest.main()
