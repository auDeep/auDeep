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

"""A meta-parser which intelligently selects a suitable parser for a data set"""
from pathlib import Path
from typing import Mapping, Optional, Sequence

from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata
from audeep.backend.parsers.cross_validated import CrossValidatedParser
from audeep.backend.parsers.dcase import DCASEParser
from audeep.backend.parsers.esc import ESCParser
from audeep.backend.parsers.partitioned import PartitionedParser
from audeep.backend.parsers.urban_sound_8k import UrbanSound8KParser


class MetaParser(LoggingMixin, Parser):
    """
    A meta-parser, which automatically selects a suitable parser for a data set.
    
    Currently, this parser chooses one of DCASEParser, PartitionedParser, CrossValidatedParser, ESCParser, and 
    UrbanSound8KParser for a data set, depending on which parser can process the data set. If more than one parser can
    process the data set, this parser can not be used, and a specific parser must be selected by the user.
    """

    def __init__(self,
                 basedir: Path):
        """
        Creates and initializes a new MetaParser for the specified data set base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._parser = None
        self._can_parse = None
        self._parsers = [
            DCASEParser(basedir),
            PartitionedParser(basedir),
            CrossValidatedParser(basedir),
            ESCParser(basedir),
            UrbanSound8KParser(basedir)
        ]

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances in the data set.
        
        This method delegates to the selected parser.
        
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
            raise IOError("unable to parse data set at {}".format(self._basedir))

        return self._parser.num_instances

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds in the data set.
        
        This method delegates to the selected parser.
        
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
            raise IOError("unable to parse data set at {}".format(self._basedir))

        return self._parser.num_folds

    @property
    def label_map(self) -> Optional[Mapping[str, float]]:
        """
        Returns the mapping of nominal labels to numeric labels for the data set.
        
        This method delegates to the selected parser.
        
        Returns
        -------
        map of str to int
            The mapping of nominal labels to numeric labels
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse data set at {}".format(self._basedir))

        return self._parser.label_map

    def can_parse(self) -> bool:
        """
        Checks whether there is exactly one parser which can parse the data set.
        
        If there is more than one parser which can parse the data set, the MetaParser does not guess a parser, but
        refuses to parse the data set.
        
        Returns
        -------
        bool
            True, if there is exactly one parser which can parse the data set, False otherwise
        """
        if self._can_parse is None:
            for p in self._parsers:
                if p.can_parse():
                    if self._parser is not None:
                        self.log.debug("multiple parsers (%s, %s) can parse data set at %s, unable to decide",
                                       self._parser.__class__.__name__, p.__class__.__name__, self._basedir)
                        self._can_parse = False
                    else:
                        self._parser = p
                        self._can_parse = True

        # noinspection PyTypeChecker
        return self._can_parse

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        This method delegates to the selected parser.
        
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
            raise IOError("unable to parse data set at {}".format(self._basedir))

        self.log.info("using %s to parse dataset at %s", self._parser.__class__.__name__, self._basedir)

        return self._parser.parse()
