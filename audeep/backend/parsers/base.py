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

"""Defines an abstract interface for data set parsers"""
import abc
from pathlib import Path
from typing import Sequence, Mapping, Optional

from audeep.backend.data.data_set import Split, Partition


class _InstanceMetadata:
    """
    Stores metadata about a single instance.
    
    This class should only be used within the audeep.backend.parsers module.
    """

    def __init__(self,
                 path: Path,
                 filename: str,
                 label_nominal: Optional[str],
                 label_numeric: Optional[int],
                 cv_folds: Optional[Sequence[Split]],
                 partition: Optional[Partition]):
        """
        Create an _InstanceMetadata object with the specified parameters.
        
        Parameters
        ----------
        path: pathlib.Path
            The absolute path to the audio file for which metadata is created
        filename: str
            The filename of the audio file
        label_nominal: str, optional
            The nominal label of the audio file
        label_numeric: int, optional
            The numeric label of the audio file
        cv_folds: list of Split, optional
            A list of cross validation splits of audio file, with one entry for each cross validation fold
        partition: Partition, optional
            The partition of the audio file
        """
        self._path = path
        self._filename = filename
        self._label_nominal = label_nominal
        self._label_numeric = label_numeric
        self._cv_folds = cv_folds
        self._partition = partition

    @property
    def path(self) -> Path:
        """
        The absolute path of the audio file for which this object stores metadata.
        
        Returns
        -------
        pathlib.Path
            The absolute path of the audio file for which this object stores metadata
        """
        return self._path

    @property
    def filename(self) -> str:
        """
        The filename of the audio file.
        
        Returns
        -------
        str
            The filename of the audio file
        """
        return self._filename

    @property
    def label_nominal(self) -> Optional[str]:
        """
        The nominal label of the audio file.
        
        Returns
        -------
        str
            The nominal label of the audio file
        """
        return self._label_nominal

    @property
    def label_numeric(self) -> Optional[int]:
        """
        The numeric label of the audio file.
        
        Returns
        -------
        int
            The numeric label of the audio file
        """
        return self._label_numeric

    @property
    def cv_folds(self) -> Optional[Sequence[Split]]:
        """
        The cross validation splits for each cross-validation fold.
        
        The returned list contains one entry for each cross-validation fold. The entry at index i in the returned list 
        specifies whether the instance belongs to the training split or the validation split in fold i. 
        
        Returns
        -------
        list of Split
            A list of cross-validation splits, with one entry for each cross-validation fold
        """
        return self._cv_folds

    @property
    def partition(self) -> Optional[Partition]:
        """
        The partition of the audio file.
        
        Returns
        -------
        Partition
            The partition of the audio file
        """
        return self._partition


class Parser:
    """
    Base class for all data set parsers.
    
    It is assumed that all data sets reside within a directory on the file system, the data set base directory, which 
    may contain arbitrary files and directories. Based on the contents of the base directory, a data set parser decides 
    whether it can parse the data set or not. If it can, it has to provide various metadata about the data set in
    general, as well as metadata about each audio file.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 basedir: Path):
        """
        Creates and initializes a new data set parser for the specified base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        if not basedir.exists():
            raise IOError("could not open dataset base directory at {}".format(basedir))

        self._basedir = basedir

    @abc.abstractmethod
    def can_parse(self) -> bool:
        """
        Returns whether this parser can parse the data set at the base directory passed to the constructor.
        
        Returns
        -------
        bool
            True, if this parser can parse the data set, False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        If the data set contains label information, returns a mapping of nominal labels to numeric label values.
        
        Returns
        -------
        map from str to int
            A mapping of nominal labels to numeric label values if the data set contains label information, None 
            otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def num_instances(self) -> int:
        """
        Returns the number of instances, i.e. audio files, in the data set
        
        Returns
        -------
        int
            The number of instances, i.e. audio files, in the data set
        """
        pass

    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """
        Returns the number of cross validation folds in the data set, or zero if there is no cross-validation setup.
        
        Returns
        -------
        int
            The number of cross validation folds in the data set
        """
        pass

    @abc.abstractmethod
    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in the data set.
        
        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in a
        deterministic order, but this order may vary between different parsers.
        
        Returns
        -------
        list of _InstanceMetadata
            Metadata for each instance
        """
        pass
