import os
import subprocess
from os.path import join as pjoin

import numpy as np
from setuptools import Extension, setup

# Force Cython until release
USE_CYTHON = True
# try:
#     from Cython.Build import cythonize
#     USE_CYTHON = True
# except:
#     USE_CYTHON = False

ext = ".pyx" if USE_CYTHON else ".c"

# -- CUDA detection helpers (optional build) --


def _find_in_path(name: str, path: str | None):
    if not path:
        return None
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def _detect_cuda_home():
    for key in ("CUDA_HOME", "CUDA_PATH", "CUDAHOME"):
        val = os.environ.get(key)
        if val and os.path.exists(val):
            return os.path.abspath(val)

    nvcc_path = _find_in_path("nvcc", os.environ.get("PATH"))
    if nvcc_path:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(nvcc_path))))

    candidates = ["/usr/local/cuda"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def locate_cuda_optional(require: bool = False):
    home = _detect_cuda_home()
    if not home:
        if require:
            raise EnvironmentError("CUDA toolkit not found. Set CUDA_HOME (or add nvcc to PATH).")
        return None

    include_dir = pjoin(home, "include")
    lib_dir = None
    for d in (pjoin(home, "lib64"), pjoin(home, "lib")):
        if os.path.exists(d):
            lib_dir = d
            break

    nvcc = _find_in_path("nvcc", os.environ.get("PATH")) or pjoin(home, "bin", "nvcc")

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": include_dir,
        "lib": lib_dir,
    }

    if not os.path.exists(cudaconfig["include"]):
        if require:
            raise EnvironmentError(f"CUDA include directory not found: {cudaconfig['include']}")
        return None

    return cudaconfig


from Cython.Distutils import build_ext


def _should_build_cuda():
    # If FORCE_CUDA is set, require CUDA to be found
    force = os.environ.get("FORCE_CUDA", "").lower() in ("1", "true", "yes")
    if force:
        return True, True
    # Default behavior: attempt build if CUDA is detectable; not required
    return True, False


class build_ext_with_cuda(build_ext):
    def build_extensions(self):
        build_cuda, require_cuda = _should_build_cuda()
        cuda_cfg = locate_cuda_optional(require=require_cuda) if build_cuda else None

        if cuda_cfg:
            os.makedirs("build/temp_cuda", exist_ok=True)
            subprocess.check_call(
                [
                    cuda_cfg["nvcc"],
                    "-c",
                    "-O2",
                    "-Xcompiler",
                    "-fPIC",
                    "src/fastlbp/cuda/cuda/main.cu",
                    "-o",
                    "build/temp_cuda/main.o",
                    "-I",
                    f"{cuda_cfg['include']}",
                ]
            )

            for ext in self.extensions:
                if getattr(ext, "name", "").endswith("_gpu") or "cuda" in ext.name:
                    ext.extra_objects = ["build/temp_cuda/main.o"]

            # Also inject CUDA include/lib directories to the matching extension
            for ext in self.extensions:
                if getattr(ext, "name", "") == "fastlbp._lbp_gpu":
                    if cuda_cfg.get("include"):
                        ext.include_dirs = list(set(list(ext.include_dirs) + [cuda_cfg["include"]]))
                    if cuda_cfg.get("lib"):
                        ext.library_dirs = list(set(list(getattr(ext, "library_dirs", [])) + [cuda_cfg["lib"]]))

        if not cuda_cfg:
            self.extensions = [e for e in self.extensions if getattr(e, "name", "") != "fastlbp._lbp_gpu"]

        super().build_extensions()


extensions = [
    Extension("fastlbp._lbp", [f"src/_lbp{ext}"], include_dirs=[np.get_include()]),
    Extension(
        "fastlbp._lbp_gpu",
        [f"src/fastlbp/cuda/_lbp_gpu{ext}"],
        libraries=["cudart"],
        language="c++",
        include_dirs=[np.get_include()],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, language_level="3str")

setup(
    name="fastlbp",
    include_dirs=[np.get_include()],
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext_with_cuda},
)
