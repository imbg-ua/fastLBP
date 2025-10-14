import os
from os.path import join as pjoin
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from icecream import ic
import numpy as np
import subprocess


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()
# ic(CUDA)


class custom_build_ext(build_ext):
    def build_extensions(self):
        # Ensure lib directory exists
        os.makedirs("build/temp_cuda", exist_ok=True)

        # Compile the CUDA file into an object file
        subprocess.check_call([
            "nvcc", "-c", "-O2", "-Xcompiler", "-fPIC",
            "cuda/main.cu",
            "-o", "build/temp_cuda/main.o",
            "-I", f"{CUDA['include']}"
        ])
        
        # Add the object file to the first extension's extra_objects
        for ext in self.extensions:
            ext.extra_objects = ["build/temp_cuda/main.o"]
        
        build_ext.build_extensions(self)

ext = Extension('lbp_cuda',
                sources=['_lbp_gpu.pyx'],
                libraries=['cudart'],  # , os.path.join(CUDA['lib64'], 'cudart')],
                language='c++',
                include_dirs=[CUDA['include'], np.get_include()],
                library_dirs=[CUDA['lib64']]
                )

setup(
    name='lbp_cuda',
    include_dirs=[CUDA['include'], np.get_include()],
    ext_modules=[ext],
    cmdclass={'build_ext': custom_build_ext},
)