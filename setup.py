#!/usr/bin/env python
import re
import sys
from subprocess import CalledProcessError, check_output

from setuptools import setup, find_packages

PROJECT = "auDeep"
VERSION = "0.9.6a1"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "maurice.gerczuk@informatik.uni-augsburg.de"
URL = "https://github.com/auDeep/auDeep"

with open("DESCRIPTION.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()
    
dependencies = [
    "cliff>=3.3, <3.4",
    "liac-arff>=2.4",
    "matplotlib>=3.2",
    "netCDF4==1.4.2",
    "liac-arff>=2.4",
    "matplotlib>=3.2",
    "netCDF4==1.4.2",
    "pandas>=1.0, <1.2",
    "pysoundfile>=0.9",
    "scipy>=1.4",
    "scikit-learn>=0.23",
    "xarray==0.10.0",
]
if sys.platform.startswith("darwin"):
    dependencies.append("tensorflow>=1.15.2,<2")
else:
    dependencies.append("tensorflow-gpu>=1.15.2,<2")

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="auDeep is a Python toolkit for unsupervised feature learning with deep neural networks (DNNs)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    platforms=["Any"],
    scripts=[],
    provides=[],
    install_requires=dependencies,
    namespace_packages=[],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "audeep = audeep.main:main"
        ],
        "audeep.commands": [
            "preprocess = audeep.cli.extract_spectrograms:ExtractSpectrograms",
            "export = audeep.cli.export:Export",
            "import = audeep.cli.import_data:Import",
            "modify = audeep.cli.modify:Modify",
            "upsample = audeep.cli.upsample:Upsample",
            "fuse = audeep.cli.fuse:FuseDataSets",
            "fuse chunks = audeep.cli.fuse:FuseChunks",
            "validate = audeep.cli.validate:Validate",
            "visualize tsne = audeep.cli.visualize:VisualizeTSNE",
            "inspect raw = audeep.cli.inspect:InspectRaw",
            "inspect netcdf = audeep.cli.inspect:InspectNetCDF",
            "t-rae train = audeep.cli.train:TrainTimeAutoencoder",
            "t-rae generate = audeep.cli.generate:GenerateTimeAutoencoder",
            "f-rae train = audeep.cli.train:TrainFrequencyAutoencoder",
            "f-rae generate = audeep.cli.generate:GenerateFrequencyAutoencoder",
            "ft-rae train = audeep.cli.train:TrainFrequencyTimeAutoencoder",
            "ft-rae generate = audeep.cli.generate:GenerateFrequencyTimeAutoencoder",
            "mlp evaluate = audeep.cli.evaluate:MLPEvaluation",
            "svm evaluate = audeep.cli.evaluate:SVMEvaluation",
            "mlp predict = audeep.cli.predict:MLPPrediction",
            "svm predict = audeep.cli.predict:SVMPrediction",
        ],
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
            
        'Environment :: GPU :: NVIDIA CUDA :: 10.0',
        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.7',
    ],
    keywords='machine-learning audio-analysis science research',
    project_urls={
        'Source': 'https://github.com/auDeep/auDeep/',
        'Tracker': 'https://github.com/auDeep/auDeep/issues',
    },
    zip_safe=False,
)
