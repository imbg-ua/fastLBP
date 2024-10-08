# This file does not depend on fastlbp_imbg lib and fastlbp.py file.

import numpy as np
from typing import Literal, Any, Union
from collections import namedtuple
from scipy.linalg import block_diag

def create_sample_image(height: int, width: int, nchannels: Literal[1,3], type: Literal['png','jpg','tiff']='tiff', dir: str='tmp'):
    """
    Create a white noise image of specified size and file type.
    Return the absolute path of this image.

    - A function does not recreate the image if it already exists.
    - Image will be saved as ./{dir}/img_{mode}_{height}x{width}.{type} where
        - `mode` is Pillow image mode: 'L' for nchannels=1 and 'RGB' for nchannels=3,
        - `type` is in ['png', 'jpg', 'tiff']
    - the only `nchannels` values allowed are 1 and 3
    """
    from PIL import Image
    import numpy as np
    import os
    import time

    t = time.perf_counter()
    assert int(height) == height
    assert int(width) == width
    assert int(nchannels) in [1,3] 
    height, width, nchannels = int(height), int(width), int(nchannels)

    mode = 'L' if nchannels == 1 else 'RGB'

    if type not in ['png', 'jpg', 'tiff']:
        raise ValueError(f"Unsupported image type: {type}. Supported types are png, jpg, tiff")
    imgname = f"{dir}/img_{mode}_{height}x{width}.{type}"

    if os.path.isfile(imgname):
        print(f"create_sample_image: {imgname} already exists")
        return os.path.abspath(imgname)
    
    print(f"create_sample_image: creating random {imgname}")
    os.makedirs(dir, exist_ok=True)
    if mode == 'L':
        image_data = np.random.rand(height, width) * 256 
    else:
        image_data = np.random.rand(height, width, 3) * 256 
    image_data = np.uint8(image_data)
    img = Image.fromarray(image_data, mode)
    img.save(imgname) # will handle file types automatically

    print(f"create_sample_image: done in {time.perf_counter()-t:.5g}s")
    return os.path.abspath(imgname)


def load_sample_image(height: int, width: int, nchannels: int, type="tiff", dir='tmp', create: bool=True):
    """
    Load (and create if needed) an image created by `create_sample_image` function.
    Return a contiguous np.ndarray of shape (height, width, nchannels) and dtype uint8.

    - The function looks for ./{dir}/img_{mode}_{height}x{width}.{type}
    - Remember to match the `dir` parameter with that passed to `create_sample_image`
    - always returns np.ndarray with 3 dimensions
    - executes `Image.MAX_IMAGE_PIXELS = None` first
    """
    from skimage.io import imread
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    if create: 
        _ = create_sample_image(height, width, nchannels, type, dir)
        
    if type == 'png' or type == 'jpg':
        print(f"Note that image type {type} requires 3x memory to read if using skimage")

    mode = 'L' if nchannels == 1 else 'RGB'
    imgname = f"{dir}/img_{mode}_{height}x{width}.{type}"
    print(f"load_sample_image: loading sample image {imgname}")
    data = imread(imgname)
    if len(data.shape) == 2:
        data = data[:,:,None]
    return data


from dataclasses import dataclass
@dataclass
class FeatureDetails:
    channel: int
    R: float
    P: int
    lbp_code: int
    feature_number: int
    label: str
    def __init__(self, channel, R, P, lbp_code, feature_number=-1, label=""):
        self.channel=channel
        self.R=R
        self.P=P
        self.lbp_code=lbp_code
        self.feature_number=feature_number
        self.label=label

def get_all_features_details(nchannels: int, radii_list: list[float], npoints_list: list[int]) -> list[FeatureDetails]:
    """
    Get a list of detailed descriptions for all features generated by run_fastlbp in the same order. 
    """
    assert len(radii_list) == len(npoints_list)
    features = []
    feature_number = 0
    for nc in range(nchannels):
        for (r,p) in zip(radii_list, npoints_list):
            for i in range(p+2):
                label = f"{feature_number}_ch{nc}_r{r}_p{p}_lbp{i}"
                features.append(FeatureDetails(nc,r,p,i,feature_number,label))
                feature_number += 1
    return features

def get_feature_details(nchannels: int, radii_list: list[float], npoints_list: list[int], feature_number: int) -> FeatureDetails:
    """
    Get a single detailed descriptions for a feature generated by run_fastlbp where `feature_number` is its index in a feature vector.

    This function calls `get_all_features_details()` under the hood, so consider using it instead.
    """
    nfeat = (np.array(npoints_list) + 2).sum()
    assert 0 <= feature_number <= nfeat
    return get_all_features_details(nchannels,radii_list,npoints_list)[feature_number]


def get_patch(data, patchsize, pr, pc):
    """
    a shorthand for `data[(pr*patchsize):((pr+1)*patchsize), (pc*patchsize):((pc+1)*patchsize)]`
    """
    return data[(pr*patchsize):((pr+1)*patchsize), (pc*patchsize):((pc+1)*patchsize)]

def patchify_image_mask(img_mask, patchsize, edit_img_mask=False, method='any'):
    """
    Convert image mask of size (h,w) to a patch mask of size (h//patchsize, w//patchsize). 
    Basically, this is a mask downscaling function.

    if method is 'any' (default):
        Patch is included if at least one its img_mask pixel is non-zero.
        If `edit_img_mask`, then fill the included patch of img_mask with ones INPLACE;
        otherwise do not change the patch.

    if method is 'all':
        Patch is included if all its img_mask pixels are non-zero.
        If `edit_img_mask`, then fill the excluded patch of img_mask with zeros INPLACE;
        otherwise do not change the patch.
    
    Ignore trailing pixels if img_mask cannot be divided in the integer number of patches.

    Returns: 
        a patch mask of size (h//patchsize, w//patchsize) with 1 for included patches and 0 for excluded ones.
    """
    assert (len(img_mask.shape) == 2) or (len(img_mask.shape) == 3 and img_mask.shape[2] == 1), "img_mask should be 2d or have a single channel"
    
    assert method in ['any', 'all'], "Mask completion methods are 'any' or 'all'"
    mask_shape = img_mask.shape
    nprows, npcols = mask_shape[0] // patchsize, mask_shape[1] // patchsize

    # patch mask is filled with ONES at first
    patch_mask = np.ones((nprows, npcols), dtype=np.uint8)
    method_all = (method == 'all')
    method_any = not method_all

    for pr in range(nprows):
        for pc in range(npcols):
            patch = get_patch(img_mask, patchsize, pr, pc)
            if (patch == 0).any():
                zeropatch = (patch == 0).all()
                if method_all or zeropatch:
                    patch_mask[pr,pc] = 0
                    if edit_img_mask:
                        patch.fill(0)
                else: # method_any and not zeropatch
                    patch_mask[pr,pc] = 1
                    if edit_img_mask:
                        patch.fill(1)
                
    return patch_mask


"""
Reduced features utils.
- Reduced Histogram Masks is a set of masks to produce a reduced histogram for a single LBP run
- Reduced Feature Masks is a block-diagonal matrix to produce a reduced feature vector for the whole multi-radial LBP features array
"""

MinimalHistMasks = namedtuple('MinimalHistMasks', 'flat, corner, edge, nonuniform')
ReducedHistMasks = namedtuple('ReducedHistMasks', 'flat_lo, corner_lo, edge, corner_hi, flat_hi, nonuniform')

def get_reduced_hist_masks(P, method='reduced', min_features_to_reduce=12):
    """
    Return an array of 4 or 6 masks or np.eye for a single histogram. 
    That is, for a single LBP run, NOT multiradial LBP features.
    Each mask correspond to a certain type of lbp codes that are worth grouping together.

    Note: no feature reduction is done for P+2 < min_features_to_reduce 

    ## Parameters:
    - P: int, npoints
    - method: str, default 'reduced'
        - 'reduced' for 6 features (flat-low, corner-low, edge, corner-high, flat-high, nonuniform)
        - 'minimal' for 4 features (flat, corner, edge, nonuniform) 

    Note: nonuniform mask is an array of zeros with a single 1 at the end.

    ## Returns:
    None or a boolean numpy array of shape (6, P+2), (4, P+2) or np.eye(P+2).  
    - if method='reduced': np.stack((flat_lo, corner_lo, edge, corner_hi, flat_hi, nonuniform))
    - if method='minimal': np.stack((flat, corner, edge, nonuniform))
    - np.eye(P+2) if P+2 < min_features_to_reduce

    
    See also `reduce_features`.
    """
    if P+2 < min_features_to_reduce: return np.eye(P+2, dtype=np.uint8)
    
    P = float(P)
    ps = np.arange(P+2)
    bins = np.floor([0, P/5, 2*P/5, 3*P/5, 4*P/5, P])

    flat_lo = (ps <= bins[1])
    corner_lo = (ps > bins[1]) & (ps <= bins[2])
    edge = (ps > bins[2]) & (ps <= bins[3])
    corner_hi = (ps > bins[3]) & (ps <= bins[4])
    flat_hi = (ps > bins[4])
    flat_hi[-1] = 0
    nonuniform = (ps == P+1)
    
    if method == 'reduced':
        hist_masks = ReducedHistMasks(flat_lo, corner_lo, edge, corner_hi, flat_hi, nonuniform)
    else:
        hist_masks = MinimalHistMasks(
            flat= flat_lo | flat_hi, corner= corner_lo | corner_hi, edge= edge, nonuniform= nonuniform
        )
    
    return np.stack(hist_masks)
    
def hist_masks_as_tuple(hist_masks_array) ->  Union[MinimalHistMasks, ReducedHistMasks, None]:
    """
    Transform feature masks numpy array to a human-friendly namedtuple.
    Returns `None` if there is no feature reduction, i.e. if hist_masks_array is np.eye(P+2).
    """

    n,m = hist_masks_array.shape
    if n==4:
        return MinimalHistMasks(*hist_masks_array)
    if n==6:
        return ReducedHistMasks(*hist_masks_array)
    if n == m:
        return None
    raise NotImplementedError()

def get_reduction_matrix(nchannels: int, npoints_list: list[int], method='reduced', min_features_to_reduce=12):
    """
    Get a matrix for full multiradial LBP feature reduction.

    ## Returns:
    reduction_matrix: np.array of shape `(new_n_features, old_n_features)`

    See also `reduce_features`.
    """
    diag_blocks = [get_reduced_hist_masks(P, method, min_features_to_reduce) for P in npoints_list] * nchannels
    return block_diag(*diag_blocks)

def reduce_features(features, hist_masks_array):
    """
    Reduce histogram using `hist_masks_array` from `get_reduced_hist_masks` or `reduction_matrix` from `get_reduction_matrix`.

    ## Parameters
    - features is np.ndarray of shape (nrows,ncols,nfeatures) or (npatches,nfeatures) or (nfeatures,)
    - hist_masks_array is np.ndarray of shape (n_new_features,nfeatures) generated by `get_reduced_hist_masks` or `get_reduced_feature_masks`

    ## Returns
    An ndarray with reduced features of shape (nrows,ncols,n_new_features) or (npatches,n_new_features) or (n_new_features,)

    Note: `reduce_histogram(hist, None) == hist`
    """
    if hist_masks_array is None: return features

    return np.tensordot(features, hist_masks_array, axes=(-1,-1))



@dataclass
class ReducedFeatureDetails(FeatureDetails): 
    lbp_code_type: str
    def __init__(self, channel, R, P, lbp_code_type, feature_number=-1, label=""):
        self.channel=channel
        self.R=R
        self.P=P
        self.lbp_code=None
        self.lbp_code_type=lbp_code_type
        self.feature_number=feature_number
        self.label=label

LBP_MINIMAL_CODE_TYPES = ( 'flat', 'corner', 'edge', 'nonuniform' )
LBP_REDUCED_CODE_TYPES = ( 'flat_lo', 'corner_lo', 'edge', 'corner_hi', 'flat_hi', 'nonuniform' )

def get_all_reduced_features_details(reduction_method:str, nchannels: int, radii_list: list[float], npoints_list: list[int], min_features_to_reduce=12) -> list[FeatureDetails]:
    """
    get_all_features_details for reduced features

    ## Parameters
    - reduction_method: str, 'redudced' or 'minimal', according to `get_reduced_hist_masks`
    - nchannels: int
    - radii_list: list[float]
    - npoints_list: list[int]
    - min_features_to_reduce: int

    """
    assert len(radii_list) == len(npoints_list)
    features = []
    feature_number = 0

    n_reduced_feats = 4 if reduction_method=='minimal' else 6
    reduced_feat_names = LBP_MINIMAL_CODE_TYPES if reduction_method=='minimal' else LBP_REDUCED_CODE_TYPES

    for nc in range(nchannels):
        for (r,p) in zip(radii_list, npoints_list):
            if p+2 < min_features_to_reduce:
                for i in range(p+2):
                    label = f"{feature_number}_ch{nc}_r{r}_p{p}_lbp{i}"
                    features.append(FeatureDetails(nc,r,p,i,feature_number,label))
                    feature_number += 1
            else:
                for i in range(n_reduced_feats):
                    label = f"{feature_number}_ch{nc}_r{r}_p{p}_lbp_{reduced_feat_names[i]}"
                    features.append(ReducedFeatureDetails(nc,r,p,reduced_feat_names[i],feature_number,label))
                    feature_number += 1
    return features
