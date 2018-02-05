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

"""Data set modification"""
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load, Partition
from audeep.backend.data.eval_tools import create_cv_setup, create_partitioning
from audeep.backend.enum_parser import EnumType


class Modify(Command):
    """
    Modify data set metadata
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep "
                                 "data model.")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="Files to which to write the modified data set. Data is stored "
                                 "in netCDF 4 format according to the auDeep data model.")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--add-cv-setup",
                           default=None,
                           type=int,
                           metavar="NUM_FOLDS",
                           help="Randomly generate NUM_FOLDS evenly sized cross-validation folds")
        group.add_argument("--add-partitioning",
                           nargs="+",
                           default=None,
                           type=EnumType(Partition),
                           metavar="PARTITIONS",
                           help="Randomly generate an evenly sized partitioning with the specified partitions")
        group.add_argument("--remove-partitioning",
                           action="store_true",
                           help="Remove partition information")
        group.add_argument("--remove-cv-setup",
                           action="store_true",
                           help="Remove cross-validation information")
        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)

        if parsed_args.add_cv_setup is not None:
            data_set = create_cv_setup(data_set, num_folds=parsed_args.add_cv_setup)
        elif parsed_args.add_partitioning is not None:
            data_set = create_partitioning(data_set, partitions=parsed_args.add_partitioning)
        elif parsed_args.remove_partitioning:
            data_set = data_set.copy()

            for index in data_set:
                data_set[index].partition = None
        elif parsed_args.remove_cv_setup:
            data_set = create_cv_setup(data_set, num_folds=0)

        data_set.save(parsed_args.output)
