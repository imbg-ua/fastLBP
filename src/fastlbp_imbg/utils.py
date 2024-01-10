# This file does not depend on fastlbp_imbg lib and fastlbp.py file.

# Create a white noise image of specified size and file type.
# Return the absolute path of this image.
#
# - A function *does not recreate* the image if it already exists.
# - Image will be saved as ./{dir}/img_{mode}_{height}x{width}.{type} where
#   - mode is Pillow image mode: 'L' for nchannels=1 and 'RGB' for nchannels=3,
#   - type is in ['png', 'jpg', 'tiff']
# - Other nchannels values are not supported.
def create_sample_image(height, width, nchannels, type='tiff', dir='tmp'):
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

# Load (and create if needed) an image created by `create_sample_image` function.
# Return a contiguous np.ndarray of shape (height, width, nchannels) and dtype uint8.
#
# - The function looks for ./{dir}/img_{mode}_{height}x{width}.{type}
# - Remember to match the `dir` parameter with that passed to `create_sample_image`
# - always returns np.ndarray with 3 dimensions
def load_sample_image(height, width, nchannels, type="png", dir='tmp', create=True):
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

def get_all_features_details(nchannels:int, radii_list:[float], npoints_list:[int]) -> [FeatureDetails]:
    features = []
    feature_number = 0
    for nc in range(nchannels):
        for (r,p) in zip(radii_list, npoints_list):
            for i in range(p+2):
                label = f"ch{nc}_r{r}_p{p}_lbp{i}"
                features.append(FeatureDetails(nc,r,p,i,feature_number,label))
                feature_number += 1
    return features

def get_feature_details(nchannels:int, radii_list:[float], npoints_list:[int], feature_number:int) -> FeatureDetails:
    assert len(radii_list) == len(npoints_list)
    nfeat = (np.array(npoints_list) + 2).sum()
    assert 0 <= feature_number <= nfeat
    return get_all_features_details(nchannels,radii_list,npoints_list)[feature_number]

