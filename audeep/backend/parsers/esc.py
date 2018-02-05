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

"""Parser for the ESC-10 and ESC-50 data sets"""
import re
from pathlib import Path
from typing import Optional, Mapping, Sequence

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata


class ESCParser(LoggingMixin, Parser):
    """
    Parser for the ESC-10 and ESC-50 data sets.
    
    This parser assumes that the data set base directory contains directories specifying the labels of instances. These
    directories must have names starting with three digits followed by the string " - " followed by the nominal class
    label. Each class directory must only contain *.ogg files. Filenames must start with the fold number (1 to 5), 
    followed by the character "-".
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new ESCParser for the specified data set base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._class_dirs = sorted([file for file in basedir.glob("*")
                                   if file.is_dir() and not file.name.startswith(".")], key=lambda x: x.name)
        self._num_instances_cache = None
        self._label_map_cache = None

    def can_parse(self) -> bool:
        """
        Checks whether the directory structure in the data set base directory can be parsed by this parser.
        
        Returns
        -------
        bool
            True, if this parser can parse the directory structure in the data set base directory
        """
        class_dir_pattern = re.compile("^\d{3} - .+")

        for class_dir in self._class_dirs:  # type: Path
            if not class_dir_pattern.fullmatch(class_dir.name):
                # class directories must start with three digits followed by ' - '
                self.log.debug("cannot parse: class directory %s does not match expected pattern 'DDD - *'",
                               class_dir.name)

                return False

            class_files = list(class_dir.glob("*"))

            for ogg_file in class_files:
                if not ogg_file.suffix == ".ogg":
                    # all files in class directories must be ogg
                    self.log.debug("cannot parse: unexpected file %s in class directory %s (only *.ogg files allowed)",
                                   ogg_file.name, class_dir.name)

                    return False

                if not ogg_file.name[0].isdigit() or not ogg_file.name[1] == "-" or not 0 < int(ogg_file.name[0]) <= 5:
                    # sound files must start with fold number followed by '-'
                    self.log.debug("cannot parse: file %s has malformed name (does not start with fold number "
                                   "followed by '-'", ogg_file.name, class_dir.name)

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
            raise IOError("unable to parse ESC dataset at {}".format(self._basedir))

        if self._num_instances_cache is None:
            self._num_instances_cache = 0

            for class_dir in self._class_dirs:
                self._num_instances_cache += len(list(class_dir.glob("*.ogg")))

        # noinspection PyTypeChecker
        return self._num_instances_cache

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds, which is five for this parser.
        
        Returns
        -------
        int
            Five
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse ESC dataset at {}".format(self._basedir))

        return 5

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns the mapping of nominal to numeric labels for this data set.
        
        Nominal labels are assigned integer indices in order of the three digits prepended to the class directories.
        For example, the first three class directories of the ESC-10 data set are '001 - Dog bark', '002 - Rain', and
        '003 - Sea waves', which would result in the label map {'Dog bark': 0, 'Rain': 1, 'Sea waves': 2}.
        
        Returns
        -------
        map of str to int
            The mapping of nominal to numeric labels for this data set
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse ESC dataset at {}".format(self._basedir))

        if self._label_map_cache is None:
            self._label_map_cache = {}

            for index, class_dir in enumerate(self._class_dirs):
                label_nominal = class_dir.name[6:]

                self._label_map_cache[label_nominal] = index

        return self._label_map_cache

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in the
        order of the class directories, and in alphabetical order within class directories.

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
            raise IOError("unable to parse ESC dataset at {}".format(self._basedir))

        meta_list = []

        for class_dir in self._class_dirs:
            label_nominal = class_dir.name[6:]

            for ogg_file in class_dir.glob("*.ogg"):
                cv_folds = [Split.TRAIN] * 5
                cv_folds[int(ogg_file.name[0]) - 1] = Split.VALID

                instance_metadata = _InstanceMetadata(path=ogg_file,
                                                      filename=str(ogg_file.name),
                                                      label_nominal=label_nominal,
                                                      label_numeric=None,
                                                      cv_folds=cv_folds,
                                                      partition=None)

                meta_list.append(instance_metadata)

        return meta_list
