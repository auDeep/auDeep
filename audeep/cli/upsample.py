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

"""Upsampling commands"""
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import Partition, load
from audeep.backend.data.upsample import upsample
from audeep.backend.enum_parser import EnumType


class Upsample(Command):
    """
    Upsample a data set or some partitions of a data set
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep "
                                 "data model.")
        parser.add_argument("--partitions",
                            default=None,
                            nargs="+",
                            type=EnumType(Partition),
                            help="Partitions which should be upsampled (TRAIN, DEVEL, or TEST). If not set, the "
                                 "entire data set will be upsampled.")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="Files to which to write the upsampled data set. Data is stored "
                                 "in netCDF 4 format according to the auDeep data model.")

        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)

        if not data_set.is_fully_labeled:
            raise ValueError("data set must be fully labeled for upsampling")
        if parsed_args.partitions is not None and not data_set.has_partition_info:
            raise ValueError("data set must have partition information for upsampling of specific partitions")

        partitions = parsed_args.partitions if parsed_args.partitions is not None else None

        upsampled_data_set = upsample(data_set, partitions)
        upsampled_data_set.save(parsed_args.output)
