#!/usr/bin/env python3
from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup


setup_args = generate_distutils_setup(
    packages=["uav_continuum_whole_body_v2"],
    package_dir={"": "src"},
)

setup(**setup_args)
