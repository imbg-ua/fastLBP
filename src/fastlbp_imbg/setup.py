from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

print("UWU !!!!!")

setup(
    ext_modules=[
        Extension(
            name="fastlbp_imbg.imbg_skimage_lbp", 
            sources=["src/fastlbp_imbg/skimage_lbp.c"],
            include_dirs=[np.get_include()]
        )
    ],
)
