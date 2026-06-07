#!/usr/bin/env python3
"""
Setup script for pylibfranka.

This setup.py is designed to be run AFTER the C++ build completes.
It installs the pre-built _pylibfranka.so as a package.
"""
import os
import sys
import glob
from setuptools import setup, find_packages
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that includes platform-specific binary extensions."""
    def has_ext_modules(self):
        return True


# Find the .so file in this directory
package_dir = os.path.dirname(os.path.abspath(__file__))
so_files = glob.glob(os.path.join(package_dir, "_pylibfranka*.so"))
so_files += glob.glob(os.path.join(package_dir, "_pylibfranka*.dylib"))

if not so_files:
    print("ERROR: _pylibfranka shared library not found.", file=sys.stderr)
    print("Please build libfranka first with ./setup_server.sh", file=sys.stderr)
    sys.exit(1)

# Get just the filename for package_data
so_filenames = [os.path.basename(f) for f in so_files]

setup(
    name="pylibfranka",
    version="0.11.0",
    description="Python bindings for libfranka (Franka Emika robot control)",
    author="Franka Emika GmbH",
    packages=["pylibfranka"],
    package_dir={"pylibfranka": "."},
    package_data={"pylibfranka": so_filenames + ["__init__.py"]},
    include_package_data=True,
    distclass=BinaryDistribution,
    python_requires=">=3.7",
    zip_safe=False,
)
