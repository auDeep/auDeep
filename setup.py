#!/usr/bin/env python
import re
from subprocess import CalledProcessError, check_output

from setuptools import setup, find_packages

PROJECT = "auDeep"
VERSION = "0.9"
LICENSE = "GPLv3+"
AUTHOR = "Michael Freitag"
AUTHOR_EMAIL = "freitagm@fim.uni-passau.de"

dependencies = [
    "cliff",
    "liac-arff",
    "matplotlib",
    "netCDF4",
    "pandas",
    "pysoundfile",
    "scipy",
    "sklearn",
    "xarray"
]

try:
    import tensorflow

    tensorflow_found = True
except ImportError:
    tensorflow_found = False

if not tensorflow_found:
    # inspired by cmake's FindCUDA
    nvcc_version_regex = re.compile("release (?P<major>[0-9]+)\\.(?P<minor>[0-9]+)")
    use_gpu = False

    try:
        output = str(check_output(["nvcc", "--version"]))
        version_string = nvcc_version_regex.search(output)

        if version_string:
            major = int(version_string.group("major"))
            minor = int(version_string.group("minor"))

            if major != 8:
                print("detected incompatible CUDA version %d.%d" % (major, minor))
            else:
                print("detected compatible CUDA version %d.%d" % (major, minor))

                use_gpu = True
        else:
            print("CUDA detected, but unable to parse version")
    except CalledProcessError:
        print("no CUDA detected")
    except Exception as e:
        print("error during CUDA detection: %s", e)

    if use_gpu:
        dependencies.append("tensorflow-gpu==1.1.0")
    else:
        dependencies.append("tensorflow==1.1.0")
else:
    print("tensorflow already installed, skipping CUDA detection")

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
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
        ],
    },
    zip_safe=False,
)
