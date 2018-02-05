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

"""An argparse argument type for Enums"""
from argparse import ArgumentTypeError
from enum import Enum
from typing import Type


class EnumType:
    """
    Performs case-insensitive parsing of string arguments to enum members.
    
    Consider, for example, an enum type with a member called MEMBER. Then, all string values with lower case 
    equivalent "member" will be parsed to the enum member MEMBER.
    """

    def __init__(self,
                 enum_class: Type[Enum]):
        """
        Create and initialize a new EnumParser for the specified Enum type.
        
        Parameters
        ----------
        enum_class: Type[Enum]
            The enum class for which values should be parsed
        """
        self._enum_class = enum_class
        self._member_map = {x.name.lower(): x.name for x in enum_class}

    def __call__(self, arg: str):
        """
        Try to parse the specified string to an enum member.
        
        Parameters
        ----------
        arg: str
            The string value which should be parsed

        Returns
        -------
        enum
            A member of the enum type passed to this class which has the specified name
            
        Raises
        ------
        argparse.ArgumentTypeError
            If the enum does not have a member with the specified name
        """
        arg_lower = arg.lower()

        if not arg_lower in self._member_map:
            raise ArgumentTypeError("invalid choice %s (choose from %s)" % (arg, ", ".join(self._member_map.values())))

        return self._enum_class[self._member_map[arg_lower]]
