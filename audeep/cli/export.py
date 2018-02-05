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

"""Export command"""
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load
from audeep.backend.data.export import ExportFormat, export
from audeep.backend.enum_parser import EnumType


class Export(Command):
    """
    Export a data set from netCDF 4 to several other formats
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep "
                                 "data model.")
        parser.add_argument("--format",
                            type=EnumType(ExportFormat),
                            required=True,
                            help="The export format (CSV, or ARFF)")
        parser.add_argument("--labels-last",
                            action="store_true",
                            help="append labels to the end of feature vectors, instead of the beginning")
        parser.add_argument("--name",
                            type=str,
                            default=None,
                            help="The name of generated files. By default, the name of the input file is used.")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="The output base directory. Partitions and cross-validation folds are written to "
                                 "separate directories.")

        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        if parsed_args.name is None:
            name = parsed_args.input.with_suffix("").name
        else:
            name = parsed_args.name

        data_set = load(parsed_args.input)
        export(basedir=parsed_args.output,
               name=name,
               data_set=data_set,
               labels_last=parsed_args.labels_last,
               fmt=parsed_args.format)
