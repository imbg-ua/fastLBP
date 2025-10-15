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

# -- CUDA detection helpers --


def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system."""

    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError("nvcc not found. Please add it to your PATH or set $CUDAHOME")
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f"The CUDA {k} path could not be located in {v}")

    return cudaconfig


from Cython.Distutils import build_ext

CUDA = locate_cuda()


class build_ext_with_cuda(build_ext):
    def build_extensions(self):

        os.makedirs("build/temp_cuda", exist_ok=True)

        # compile CUDA module
        subprocess.check_call(
            [
                "nvcc",
                "-c",
                "-O2",
                "-Xcompiler",
                "-fPIC",
                "src/fastlbp/cuda/cuda/main.cu",
                "-o",
                "build/temp_cuda/main.o",
                "-I",
                f"{CUDA['include']}",
            ]
        )

        # Add the object file to the CUDA extension's extra_objects
        for ext in self.extensions:
            if getattr(ext, "name", "").endswith("_gpu") or "cuda" in ext.name:
                ext.extra_objects = ["build/temp_cuda/main.o"]

        super().build_extensions()


extensions = [
    Extension("fastlbp._lbp", [f"src/_lbp{ext}"], include_dirs=[np.get_include()]),
    # CUDA LBP extension
    Extension(
        "fastlbp._lbp_gpu",
        [f"src/fastlbp/cuda/_lbp_gpu{ext}"],
        libraries=["cudart"],  # , os.path.join(CUDA['lib64'], 'cudart')],
        language="c++",
        include_dirs=[CUDA["include"], np.get_include()],
        library_dirs=[CUDA["lib64"]],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, language_level="3str")

setup(
    name="fastlbp",
    include_dirs=[CUDA["include"], np.get_include()],
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext_with_cuda},
)
