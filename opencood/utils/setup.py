import os
from pathlib import Path

from Cython.Build import cythonize
import numpy
from setuptools import setup


# Cython and --inplace resolve paths from the current working directory.
# Anchor them to OpenCOOD so this script behaves the same from any caller cwd.
OPENCOOD_ROOT = Path(__file__).resolve().parents[2]
os.chdir(OPENCOOD_ROOT)

setup(
    name='box overlaps',
    ext_modules=cythonize('opencood/utils/box_overlaps.pyx'),
    include_dirs=[numpy.get_include()]
)
