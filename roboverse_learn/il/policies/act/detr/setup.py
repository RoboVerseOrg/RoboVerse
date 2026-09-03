# Copyright (c) 2023 Tony Z. Zhao
# SPDX-License-Identifier: MIT
#
# Adapted from ACT (https://github.com/tonyzhaozh/act), file detr/setup.py. This file is ACT's
# own; it has no counterpart in DETR.
# Changes: none (vendored verbatim; trailing newline added).
# Full license: roboverse_learn/il/policies/act/LICENSE

from distutils.core import setup

from setuptools import find_packages

setup(
    name="detr",
    version="0.0.0",
    packages=find_packages(),
    license="MIT License",
    long_description=open("README.md").read(),
)
