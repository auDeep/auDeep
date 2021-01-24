# Copyright (C) 2021 Shahin Amiriparian, Michael Freitag, Maurice Gerczuk, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Parser for the ComParE 2021 challenge tasks"""
import abc
from pathlib import Path
from typing import Optional, Mapping, Sequence

import pandas as pd
from pandas.io.parsers import read_csv

from audeep.backend.data.data_set import Partition
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata

class Compare21Parser(LoggingMixin, Parser):

    def __init__(self, basedir: Path):
        super().__init__(basedir)

        self._metadata_cache = None
        self._audio_dir = basedir / "wav"

    @abc.abstractmethod
    def label_key(self) -> str:
        pass

    def _metadata(self) -> pd.DataFrame:
        if not self.can_parse():
            raise IOError("unable to parse the ComParE 2020 Mask dataset at {}".format(self._basedir))
        if self._metadata_cache is None:
            metadata_file_train = self._basedir / "lab" / "train.csv"
            metadata_file_devel = self._basedir / "lab" / "devel.csv"
            metadata_file_test = self._basedir / "lab" / "test.csv"
            self._metadata_cache = pd.concat([pd.read_csv(metadata_file_train, sep=","), pd.read_csv(metadata_file_devel, sep=","), pd.read_csv(metadata_file_test, sep=",")]) 

        return self._metadata_cache

    def can_parse(self) -> bool:
        metadata_file_train = self._basedir / "lab" / "train.csv"
        metadata_file_devel = self._basedir / "lab" / "devel.csv"
        metadata_file_test = self._basedir / "lab" / "test.csv"
        
        if not self._audio_dir.exists():
            self.log.debug("cannot parse: audio directory at %s missing", self._audio_dir)
            return False

        if not metadata_file_train.exists(): 
            self.log.debug("cannot parse: metadata file at %s missing", metadata_file_train)
            return False

        elif not metadata_file_devel.exists():
            self.log.debug("cannot parse: metadata file at %s missing", metadata_file_devel)
            return False

        elif not metadata_file_test.exists():
            self.log.debug("cannot parse: metadata file at %s missing", metadata_file_test)
            return False
        return True

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        if not self.can_parse():
            raise IOError("inable to parse the ComParE 2021 dataset at {}".format(self._basedir))
        metadata_cache = self._metadata()
        labels = sorted(map(str, set(metadata_cache.label.values)))
        if "?" in labels:
            labels.remove("?")
        label_map = {label: i for i, label in enumerate(labels)}
        return label_map

    @property
    def num_instances(self) -> int:
        if not self.can_parse():
            raise IOError("unable to parse the ComParE 2021 dataset at {}".format(self._basedir))

        # test instances are not contained in label tsv file
        return len(list(self._audio_dir.glob("*.*")))

    @property
    def num_folds(self) -> int:
        if not self.can_parse():
            raise IOError("unable to parse the ComParE 2021 dataset at {}".format(self._basedir))

        return 0

    def parse(self) -> Sequence[_InstanceMetadata]:
        if not self.can_parse():
            raise IOError("unable to parse the ComParE 2021 dataset at {}".format(self._basedir))

        meta_list = []

        metadata = self._metadata()

        for file in sorted(self._audio_dir.glob("*.*")):
            label_nominal = metadata.loc[metadata["filename"] == file.name]["label"]

            # test labels are '?'
            if all(l != '?' for l in label_nominal):
                label_nominal = str(label_nominal.iloc[0])
            else:
                label_nominal = None

            instance_metadata = _InstanceMetadata(
                path=file,
                filename=file.name,
                label_nominal=label_nominal,
                label_numeric=None,  # inferred from label map
                cv_folds=[],
                partition=Partition.TRAIN if file.name.startswith("train") else Partition.DEVEL if file.name.startswith(
                    "devel") else Partition.TEST
            )

            self.log.debug("parsed instance %s: label = %s", file.name, label_nominal)
            meta_list.append(instance_metadata)

        return meta_list