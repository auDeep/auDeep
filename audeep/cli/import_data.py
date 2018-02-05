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

"""Data set import"""
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.import_data import DataImporter


class Import(Command):
    """
    Import a data set from CSV/ARFF to netCDF 4
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--basedir",
                            type=Path,
                            required=True,
                            help="The import base directory")
        parser.add_argument("--name",
                            type=str,
                            required=True,
                            help="The name of the data files to import")
        parser.add_argument("--filename-attribute",
                            default="filename",
                            help="The name of the filename attribute in the data files (default 'filename')")
        parser.add_argument("--chunk-nr-attribute",
                            default="chunk_nr",
                            help="The name of the chunk number attribute in the data files (default 'chunk_nr')")
        parser.add_argument("--label-nominal-attribute",
                            default="label_nominal",
                            help="The name of the nominal label attribute in the data files (default 'label_nominal')")
        parser.add_argument("--label-numeric-attribute",
                            default="label_numeric",
                            help="The name of the numeric label attribute in the data files (default 'label_numeric')")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="The output filename. Data is stored in netCDF 4 format according to the auDeep "
                                 "data model.")
        return parser

    def take_action(self, parsed_args):
        importer = DataImporter(filename_attribute=parsed_args.filename_attribute,
                                chunk_nr_attribute=parsed_args.chunk_nr_attribute,
                                label_nominal_attribute=parsed_args.label_nominal_attribute,
                                label_numeric_attribute=parsed_args.label_numeric_attribute)
        data_set = importer.import_data(basedir=parsed_args.basedir,
                                        name=parsed_args.name)
        data_set.save(parsed_args.output)
