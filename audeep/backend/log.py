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

"""Mixin class to add a log attribute"""
import logging


class LoggingMixin:
    """
    A logging mixin, which adds a log attribute to a class.
    """

    def __init__(self, *args, **kwargs):
        """
        Create and initialize the LoggingMixin.
        
        Any parameters passed to this constructor will be passed to the base class unchanged.
        
        Parameters
        ----------
        args
            Positional arguments
        kwargs
            Keyword arguments
        """
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)

        self.log = logging.getLogger(self.__class__.__name__)
