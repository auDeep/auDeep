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

"""A parser for data sets with cross validation information and labels represented by the directory structure"""
from pathlib import Path
from typing import Optional, Mapping, Sequence, List

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata


class CrossValidatedParser(LoggingMixin, Parser):
    """
    A generic parser for data sets which have cross-validation information and labels represented by the directory
    structure.
    
    This parser assumes that the data set base directory contains one directory for each cross-validation fold, named
    'fold_N' where N is the fold number starting with 1. Within each fold directory, there must be one directory for 
    each class, with the directory names specifying the nominal labels of the audio files contained in the respective 
    class directory.
    """

    def __init__(self, basedir: Path):
        """
        Creates a new CrossValidatedParser for the specified base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._dirs = list(self._basedir.glob("fold_*"))
        self._label_map_cache = None
        self._num_instances_cache = None

    def can_parse(self) -> bool:
        """
        Checks whether the directory structure in the data set base directory can be parsed by this parser.
        
        Returns
        -------
        bool
            True, if this parser can parse the directory structure in the data set base directory
        """
        if len(self._dirs) < 2:
            self.log.debug("cannot parse: need at least two fold directories")

            return False

        for dir in self._dirs:
            fold_number_str = dir.name[5:]

            if not fold_number_str.isdigit():
                self.log.debug("cannot parse: malformed fold number %s for directory %s", fold_number_str, dir)

                return False

        fold_numbers = [int(dir.name[5:]) for dir in self._dirs]

        if not set(fold_numbers) == set(range(1, len(fold_numbers) + 1)):
            self.log.debug("cannot parse: fold numbers must be [1-%d], are %s", len(fold_numbers), sorted(fold_numbers))

            return False

        for fold_dir in self._dirs:
            if not all([file.is_dir() for file in fold_dir.glob("*")]):
                self.log.warning("fold directory %s contains files that are not directories", fold_dir)

            for class_dir in (file for file in fold_dir.glob("*") if file.is_dir()):
                files = list(class_dir.glob("*.*"))  # type: List[Path]

                if len(files) == 0:
                    self.log.debug("cannot parse: class directory %s must contain at least one audio file", class_dir)

                    return False

        return True

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds in the data set.
        
        The number of cross-validation folds is equal to the number of fold directories in the data set base directory.
        
        Returns
        -------
        int
            The number of cross-validation folds in the data set
            
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse dataset at {}".format(self._basedir))

        return len(self._dirs)

    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns the mapping of nominal to numeric labels for this data set.
        
        Nominal labels are assigned integer indices in alphabetical order. For example, for a data set with nominal 
        labels 'a' and 'b', the returned label map would be {'a': 0, 'b': 1}.
        
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
            raise IOError("unable to parse dataset at {}".format(self._basedir))

        if self._label_map_cache is None:
            classes = []

            for fold_dir in self._dirs:
                class_dirs = fold_dir.glob("*")

                fold_classes = [class_dir.name for class_dir in class_dirs if class_dir.is_dir()]

                if classes and set(classes) != set(fold_classes):
                    self.log.warning("classes differ between folds")

                classes += fold_classes

            classes = set(classes)

            self._label_map_cache = dict(list(zip(sorted(classes), range(len(classes)))))

        return self._label_map_cache

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances, i.e. audio files, in this data set.
        
        Returns
        -------
        int
            The number of instances in this data set
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse dataset at {}".format(self._basedir))

        if self._num_instances_cache is None:
            self._num_instances_cache = 0

            for fold_dir in self._dirs:
                for class_dir in fold_dir.glob("*"):
                    self._num_instances_cache += len(list(class_dir.glob("*.*")))

        # noinspection PyTypeChecker
        return self._num_instances_cache

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.
        
        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in order
        of increasing fold number, and in alphabetical order by nominal label and filename within folds.
        
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
            raise IOError("unable to parse dataset at {}".format(self._basedir))

        meta_list = []

        for fold in range(self.num_folds):
            fold_dir = self._basedir / ("fold_%d" % (fold + 1))

            cv_folds = [Split.TRAIN] * self.num_folds
            cv_folds[fold] = Split.VALID

            for class_dir in sorted(fold_dir.glob("*")):
                for file in sorted(class_dir.glob("*.*")):
                    filename = "{}/{}/{}".format(fold_dir.name, class_dir.name, file.name)

                    instance_metadata = _InstanceMetadata(path=file,
                                                          filename=filename,
                                                          label_nominal=class_dir.name,
                                                          label_numeric=None,
                                                          cv_folds=cv_folds,
                                                          partition=None)

                    self.log.debug("parsed instance %s: label = %s", filename, class_dir.name)
                    meta_list.append(instance_metadata)

        return meta_list
