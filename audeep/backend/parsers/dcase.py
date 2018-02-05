# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
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

"""Parser for the development partition of the DCASE 2017 acoustic scene classification data set"""
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata

_DCASE_LABEL_MAP = {
    "beach": 0,
    "bus": 1,
    "cafe/restaurant": 2,
    "car": 3,
    "city_center": 4,
    "forest_path": 5,
    "grocery_store": 6,
    "home": 7,
    "library": 8,
    "metro_station": 9,
    "office": 10,
    "park": 11,
    "residential_area": 12,
    "train": 13,
    "tram": 14
}


class DCASEParser(LoggingMixin, Parser):
    """
    Parser for the DCASE 2017 Acoustic Scene Classification development data set.
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new DCASEParser for the specified base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._metadata_cache = None
        self._cv_setup_cache = None

    def _metadata(self) -> pd.DataFrame:
        """
        Read the meta.txt file in the data set base directory containing general data set metadata.
        
        The meta.txt file is read only once and cached.
        
        Returns
        -------
        pandas.DataFrame
            The metadata contained in the meta.txt file as a pandas DataFrame
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse DCASE dataset at {}".format(self._basedir))
        if self._metadata_cache is None:
            self._metadata_cache = pd.read_table(str(self._basedir / "meta.txt"), header=None)

        # noinspection PyTypeChecker
        return self._metadata_cache

    def _cv_setup(self) -> Sequence[pd.DataFrame]:
        """
        Reads the evaluation_setup/fold*_train.txt files containing the cross-validation setup.
        
        The evaluation_setup/fold*_train.txt files are read only once and cached.
        
        
        Returns
        -------
        list of pandas.DataFrame
            A list of pandas DataFrames containing the contents of the evaluation_setup/fold*_train.txt files,
            with one entry per cross-validation fold
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse DCASE dataset at {}".format(self._basedir))
        if self._cv_setup_cache is None:
            self._cv_setup_cache = [
                pd.read_table(str(self._basedir / "evaluation_setup" / ("fold%d_train.txt" % (i + 1))),
                              header=None) for i in range(4)]

        return self._cv_setup_cache

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances in the data set.
        
        Returns
        -------
        int
            The number of instances in the data set
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return len(self._metadata())

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds, which is four for this parser.
        
        Returns
        -------
        int
            Four
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return 4

    @property
    def label_map(self) -> Mapping[str, int]:
        """
        Returns the mapping of nominal to numeric labels.
        
        Nominal labels are assigned integer indices in alphabetical order. That is, the following label map is returned:
        
        "beach": 0,
        "bus": 1,
        "cafe/restaurant": 2,
        "car": 3,
        "city_center": 4,
        "forest_path": 5,
        "grocery_store": 6,
        "home": 7,
        "library": 8,
        "metro_station": 9,
        "office": 10,
        "park": 11,
        "residential_area": 12,
        "train": 13,
        "tram": 14
        
        Returns
        -------
        map of str to int
            The mapping of nominal to numeric labels
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return _DCASE_LABEL_MAP

    def can_parse(self) -> bool:
        """
        Checks whether the data set base directory contains the DCASE 2017 Acoustic Scene Classification development 
        data set.
        
        Currently, this method checks for the presence of a meta.txt file in the data set base directory, and the
        presence of evaluation_setup/foldN_train.txt files for N in [1, 2, 3, 4].
        
        Returns
        -------
        bool
             True, if the data set base directory contains the DCASE 2017 Acoustic Scene Classification development 
                 data set, False otherwise
        """
        meta_txt_exists = (self._basedir / "meta.txt").exists()
        cv_setup_exists = [(self._basedir / "evaluation_setup" / ("fold%d_train.txt" % (fold + 1))).exists()
                           for fold in range(self.num_folds)]

        return meta_txt_exists and all(cv_setup_exists)

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in the
        order in which they appear in the meta.txt file.

        Returns
        -------
        list of _InstanceMetadata
            A list of _InstanceMetadata containing one entry for each parsed audio file
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse DCASE dataset at {}".format(self._basedir))

        metadata = self._metadata()
        cv_setup = self._cv_setup()

        meta_list = []

        for index, row in metadata.iterrows():
            filename = row[0]
            label_nominal = row[1]

            if label_nominal not in _DCASE_LABEL_MAP:
                raise IOError("invalid label for DCASE data: {}".format(label_nominal))

            cv_folds = []

            for fold_metadata in cv_setup:
                cv_folds.append(Split.TRAIN if filename in fold_metadata.iloc[:, 0].values else Split.VALID)

            instance_metadata = _InstanceMetadata(path=self._basedir / filename,
                                                  filename=str(Path(filename).name),
                                                  label_nominal=label_nominal,
                                                  label_numeric=None,
                                                  cv_folds=cv_folds,
                                                  partition=None)

            self.log.debug("parsed instance %s: label = %s", filename, label_nominal)
            meta_list.append(instance_metadata)

        return meta_list
