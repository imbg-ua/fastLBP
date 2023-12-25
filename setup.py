from setuptools import Extension, setup
import numpy

# Force Cython until release
USE_CYTHON = True
# try:
#     from Cython.Build import cythonize
#     USE_CYTHON = True
# except:
#     USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        "fastlbp_imbg.lbp", ["src/lbp"+ext],
        include_dirs=[numpy.get_include()]
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, language_level='3str')

setup(
    ext_modules=extensions
)


