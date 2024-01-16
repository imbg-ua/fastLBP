import numpy as np
from ..utils import (
    get_patch,
    complete_background_mask,
    load_sample_image
)
from ..lbp import (
    uniform_lbp_uint8,
    uniform_lbp_uint8_masked,
    uniform_lbp_uint8_patch_masked,
)
import unittest

class TestFastlbpUtils(unittest.TestCase):
    def test_pixel_mask_allones(self):
        image = load_sample_image(512,512,1,'tiff',create=True)[:,:,0]

        A1 = uniform_lbp_uint8(image, 7, 1)
        A2 = uniform_lbp_uint8(image, 13, 2)
        A3 = uniform_lbp_uint8(image, 25, 3)

        allones_mask = np.ones_like(image, dtype=np.uint8)
        B1 = uniform_lbp_uint8_masked(image, allones_mask, 7, 1)
        B2 = uniform_lbp_uint8_masked(image, allones_mask, 13, 2)
        B3 = uniform_lbp_uint8_masked(image, allones_mask, 25, 3)

        self.assertTrue((A1==B1).all())
        self.assertTrue((A2==B2).all())
        self.assertTrue((A3==B3).all())
    
    def test_patch_mask_allones(self):
        image = load_sample_image(512,512,1,'tiff',create=True)[:,:,0]

        A1 = uniform_lbp_uint8(image, 7, 1)
        A2 = uniform_lbp_uint8(image, 13, 2)
        A3 = uniform_lbp_uint8(image, 25, 3)

        ps = 64
        nprows, npcols = image.shape[0]//ps, image.shape[1]//ps
        allones_mask = np.ones((nprows, npcols), dtype=np.uint8)
        B1 = uniform_lbp_uint8_patch_masked(image, allones_mask, ps, 7, 1)
        B2 = uniform_lbp_uint8_patch_masked(image, allones_mask, ps, 13, 2)
        B3 = uniform_lbp_uint8_patch_masked(image, allones_mask, ps, 25, 3)

        self.assertTrue((A1==B1).all())
        self.assertTrue((A2==B2).all())
        self.assertTrue((A3==B3).all())

    def test_patch_mask_exclude(self):
        noise = load_sample_image(512,512,1,'tiff',create=True)[:,:,0]
        x = np.arange(-256,256)
        assert x.shape == (512,)
        xx,yy = np.meshgrid(x,x,sparse=True)
        disk = (xx*xx + yy*yy) < 200*200
        image = noise * disk
        mask = disk.copy().astype(np.uint8)
        assert mask.shape == (512,512)

        patchsize = 32
        patch_mask = complete_background_mask(mask, patchsize, edit_img_mask=True, method='exclude')
        
        B1 = uniform_lbp_uint8_masked(image, mask, 7, 1)
        B2 = uniform_lbp_uint8_masked(image, mask, 13, 2)
        B3 = uniform_lbp_uint8_masked(image, mask, 25, 3)
        
        C1 = uniform_lbp_uint8_patch_masked(image, patch_mask, patchsize, 7, 1)
        C2 = uniform_lbp_uint8_patch_masked(image, patch_mask, patchsize, 13, 2)
        C3 = uniform_lbp_uint8_patch_masked(image, patch_mask, patchsize, 25, 3)
        
        self.assertTrue((C1==B1).all())
        self.assertTrue((C2==B2).all())
        self.assertTrue((C3==B3).all())
