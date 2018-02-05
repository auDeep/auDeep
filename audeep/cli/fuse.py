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

"""Fusion commands"""
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load, concat_features, concat_chunks
from audeep.backend.log import LoggingMixin


class FuseDataSets(LoggingMixin, Command):
    """
    Fuse different data sets along feature dimensions
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            nargs="+",
                            type=Path,
                            required=True,
                            help="Files containing data sets in netCDF 4 format, conformant to the auDeep"
                                 "data model. Data sets must have identical metadata.")
        parser.add_argument("--dimension",
                            default="generated",
                            help="Dimension to fuse along. Defaults to 'generated', which is the dimension produced by "
                                 "the 'rae generate' command. Use the 'inspect' command to check dimension names of a "
                                 "data set.")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="The output filename. Data is stored in netCDF 4 format according to the auDeep "
                                 "data model.")

        return parser

    def take_action(self, parsed_args):
        for file in parsed_args.input:
            if not file.exists():
                raise IOError("failed to open data set at {}".format(file))

        data_sets = [load(file) for file in parsed_args.input]

        self.log.info("fusing %d data sets along dimension '%s'", len(parsed_args.input), parsed_args.dimension)

        result = concat_features(data_sets, parsed_args.dimension)
        result.save(parsed_args.output)


class FuseChunks(LoggingMixin, Command):
    """
    Fuse chunks along feature dimensions
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep data "
                                 "model")
        parser.add_argument("--dimension",
                            default="generated",
                            help="Dimension to fuse along. Defaults to 'generated', which is the dimension produced by "
                                 "the 'rae generate' command. Use the 'inspect' command to check dimension names of a "
                                 "data set.")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="The output filename. Data is stored in netCDF 4 format according to the auDeep "
                                 "data model.")

        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)

        self.log.info("fusing chunks along dimension '%s'", parsed_args.dimension)

        data_set = concat_chunks(data_set, parsed_args.dimension)

        data_set.save(parsed_args.output)
