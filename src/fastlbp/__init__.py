from . import _lbp, lbp, utils
from .fastlbp import (
    FastlbpResult,
    get_p_for_r,
    get_radii,
    run_chunked_fastlbp,
    run_fastlbp,
    run_patch_fastlbp,
)
from .utils import (
    create_sample_image,
    get_all_features_details,
    get_feature_details,
    load_sample_image,
    patchify_image_mask,
)

__all__ = [
    "run_fastlbp",
    "run_chunked_fastlbp",
    "run_patch_fastlbp",
    "FastlbpResult",
    "get_radii",
    "get_p_for_r",
    "load_sample_image",
    "create_sample_image",
    "get_all_features_details",
    "get_feature_details",
    "patchify_image_mask",
    "lbp",
    "_lbp",
    "utils",
]

# Conditionally expose CUDA entrypoint if the extension is available
try:
    # Import the compiled GPU extension to ensure GPU build is present
    from ._lbp_gpu import cuda_lbp  # noqa: F401

    # If import succeeds, expose the high-level CUDA pipeline
    from .fastlbp import run_cuda_fastlbp

    __all__.append("run_cuda_fastlbp")
except Exception:
    # GPU extension not available; keep CPU-only API
    pass

# I will use the following versioning scheme https://stackoverflow.com/a/76129798
# - main branch gets 1.2.3
# - dev branch gets 1.2.3.dev1
# - feature branch gets 1.2.3.dev0+feature.improve.logs
__version__ = "0.3.3"
