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

"""A parser which does not parse any metadata"""
from pathlib import Path
from typing import Optional, Mapping, Sequence

from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata


class NoMetadataParser(LoggingMixin, Parser):
    """
    Parser which does not parse any metadata besides the filenames of instances.
    
    As a result, this parser can process any directory structure containing audio files. It is useful for cases in which
    only the basic feature learning capabilities of the auDeep application are required.
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new NoMetadataParser for the specified data set base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._num_instances_cache = None

    def can_parse(self) -> bool:
        """
        Returns True, since this parser can process any directory structure.
        
        Returns
        -------
        bool
            True
        """
        return True

    @property
    def num_instances(self) -> int:
        """
        Returns the number instances in the data set.
        
        Simply counts the number of WAV files anywhere below the data set base directory.
        
        Returns
        -------
        int
            The number instances in the data set
        """
        if self._num_instances_cache is None:
            self._num_instances_cache = len(list(self._basedir.rglob("*.wav")))

        # noinspection PyTypeChecker
        return self._num_instances_cache

    @property
    def num_folds(self) -> int:
        """
        Returns zero, since this parser does not parse cross-validation information.
        
        Returns
        -------
        int
            Zero
        """
        return 0

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns None, since this parser does not parse labels.
        
        Returns
        -------
        map of str to int
            None
        """
        return None

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.
        
        Returns
        -------
        list of _InstanceMetadata
            A list of _InstanceMetadata containing one entry for each parsed audio file
        """
        meta_list = []

        for file in self._basedir.rglob("*.wav"):  # type: Path
            filename = str(file.relative_to(self._basedir))

            instance_metadata = _InstanceMetadata(path=file,
                                                  filename=filename,
                                                  label_nominal=None,
                                                  label_numeric=None,
                                                  cv_folds=[],
                                                  partition=None)

            self.log.debug("parsed instance %s", filename)
            meta_list.append(instance_metadata)

        return meta_list
