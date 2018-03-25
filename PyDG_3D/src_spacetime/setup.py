from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy
Options.buffer_max_dims = 9
extensions = [Extension(
                "cython_euler",
                sources=["cython_euler.pyx"],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],include_dirs=[numpy.get_include()]
            )]

setup(
    ext_modules = cythonize(extensions)
)
