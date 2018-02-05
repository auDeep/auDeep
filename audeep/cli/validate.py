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

"""Data set validation command"""
import math
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load, check_integrity


class Validate(Command):
    """
    Validate integrity of a data set
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep data "
                                 "model")
        parser.add_argument("--detailed",
                            action="store_true",
                            help="Print detailed information about integrity constraint violations")

        return parser

    def _build_header_pattern(self,
                              num_digits: int):
        return "%%s: %%%d violations" % num_digits

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)
        constraint_violations = check_integrity(data_set)

        if constraint_violations is None:
            print("PASS validation")
        else:
            num_digits = int(math.ceil(math.log10(data_set.num_instances)))

            constraint_headers = [
                "Instances with the same filename must have the same nominal labels:",
                "Instances with the same filename must have the same numeric labels:",
                "Instances with the same filename must have the same cross validation information:",
                "Instances with the same filename must have the same partition:",
                "If a label map is present, all labels must conform to this map:",
                "For each filename, there must be the same number of chunks:",
                "For each filename, chunk numbers must be [0, 1, ..., num_chunks]:"
            ]

            print("FAIL validation")
            print()

            for index, header in enumerate(constraint_headers):
                print("%d. %-81s %{}d violations".format(num_digits) % (
                    index + 1, header, len(constraint_violations[index])))

                if parsed_args.detailed:
                    for violation in constraint_violations[index]:
                        print("\t%s" % violation)
