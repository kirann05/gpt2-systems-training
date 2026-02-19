from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "dataloader",
        ["DataLoader.cpp", "cnpy/cnpy.cpp"],
        include_dirs=[
            pybind11.get_include(),  # Include path for pybind11
            "/usr/include/eigen3",   # Include path for Eigen
            "./cnpy"                 # Include path for cnpy
        ],
        cxx_std=14,
        extra_compile_args=["-g"],  # Enable debugging information
        extra_link_args=["-lz"],    # Link with zlib
    ),
]

setup(
    name="dataloader",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
