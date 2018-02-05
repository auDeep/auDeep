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

"""A parser for the UrbanSound8K data set"""
from pathlib import Path
from typing import Optional, Mapping, Sequence

import pandas as pd

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata


class _ColumnNames:
    """
    Names of the metadata columns in the UrbanSound8K metadata CSV file.
    """
    FILENAME = "slice_file_name"
    CV_FOLD = "fold"
    LABEL_NUMERIC = "classID"
    LABEL_NOMINAL = "class"


class UrbanSound8KParser(LoggingMixin, Parser):
    """
    Parser for the UrbanSound8K data set.
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new UrbanSound8KParser for the specified data set base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._metadata_cache = None
        self._label_map_cache = None

    def _metadata(self) -> pd.DataFrame:
        """
        Reads the metadata/UrbanSound8K.csv metadata file.
        
        The file is read once and cached.
        
        Returns
        -------
        pandas.DataFrame
            The contents of the metadata/UrbanSound8K.csv metadata file as a pandas DataFrame
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse UrbanSound8K dataset at {}".format(self._basedir))
        if self._metadata_cache is None:
            self._metadata_cache = pd.read_csv(self._basedir / "metadata" / "UrbanSound8K.csv")

        # noinspection PyTypeChecker
        return self._metadata_cache

    def can_parse(self) -> bool:
        """
        Checks whether the data set base directory contains the UrbanSound8K data set.
        
        Currently, this method checks whether the metadata/UrbanSound8K.csv file exists, and whether the audio directory
        exists within the data set base directory.
        
        Returns
        -------
        bool
            True, if the data set base directory contains the UrbanSound8K data set, False otherwise
        """
        audio_dir = self._basedir / "audio"
        metadata_file = self._basedir / "metadata" / "UrbanSound8K.csv"

        if not audio_dir.exists():
            self.log.debug("cannot parse: audio directory at %s missing", audio_dir)

            return False

        if not metadata_file.exists():
            self.log.debug("cannot parse: metadata file at %s missing", metadata_file)

            return False

        return True

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
        if not self.can_parse():
            raise IOError("Unable to parse UrbanSound8K data set at {}".format(self._basedir))

        return len(self._metadata())

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds, which is ten for this parser.
        
        Returns
        -------
        int
            ten
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("Unable to parse UrbanSound8K data set at {}".format(self._basedir))

        return 10

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns the mapping of nominal to numeric labels.
        
        The UrbanSound8K specifies a custom mapping, which is returned by this method.
        
        Returns
        -------
        map of str to int
            The mapping of nominal to numeric labels
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("Unable to parse UrbanSound8K data set at {}".format(self._basedir))

        if self._label_map_cache is None:
            self._label_map_cache = {}

            for _, row in self._metadata().iterrows():
                label_nominal = row.loc[_ColumnNames.LABEL_NOMINAL]
                label_numeric = row.loc[_ColumnNames.LABEL_NUMERIC]

                if label_nominal not in self._label_map_cache:
                    self._label_map_cache[label_nominal] = label_numeric
                elif self._label_map_cache[label_nominal] != label_numeric:
                    raise IOError("inconsistent labels: %s has numeric values %d and %d", label_nominal, label_numeric,
                                  self._label_map_cache[label_nominal])

        return self._label_map_cache

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in the
        order in which they appear in the metadata/UrbanSound8K.csv file.

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
            raise IOError("Unable to parse UrbanSound8K data set at {}".format(self._basedir))

        meta_list = []

        for _, row in self._metadata().iterrows():
            filename = row.loc[_ColumnNames.FILENAME]
            path = self._basedir / "audio" / ("fold%d" % row.loc[_ColumnNames.CV_FOLD]) / filename

            cv_folds = [Split.TRAIN] * 10
            cv_folds[row.loc[_ColumnNames.CV_FOLD] - 1] = Split.VALID

            label_nominal = row.loc[_ColumnNames.LABEL_NOMINAL]

            instance_metadata = _InstanceMetadata(path=path,
                                                  filename=filename,
                                                  label_nominal=label_nominal,
                                                  label_numeric=None,
                                                  cv_folds=cv_folds,
                                                  partition=None)

            self.log.debug("parsed instance %s: label = %s", filename, label_nominal)
            meta_list.append(instance_metadata)

        return meta_list
