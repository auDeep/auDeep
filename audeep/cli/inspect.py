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

"""Data set inspection command"""
import importlib
import math
from pathlib import Path
from os.path import splitext, basename

import numpy as np
from cliff.command import Command
from soundfile import SoundFile, SEEK_END

from audeep.backend.data.data_set import load, Split
from audeep.backend.formatters import TableFormatter
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser


class InspectRaw(LoggingMixin, Command):
    """
    Display information about a data set that has not yet been imported
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--basedir",
                            type=Path,
                            required=True,
                            help="The data set base directory")
        parser.add_argument("--parser",
                            type=str,
                            default="audeep.backend.parsers.meta.MetaParser",
                            help="Parser for the data set file structure. Defaults to "
                                 "audeep.backend.parsers.meta.MetaParser, which supports several common file "
                                 "structures.")

        return parser

    def take_action(self, parsed_args):
        if ":" in parsed_args.parser:
            self.log.info(f'Using custom external parser: {parsed_args.parser}')
            path, class_name = parsed_args.parser.split(':')
            module_name = f'audeep.backend.parsers.custom.{splitext(basename(path))[0]}'
            spec = importlib.util.spec_from_file_location(module_name, path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            parser_class = getattr(foo, class_name)
        else:
            module_name, class_name = parsed_args.parser.rsplit(".", 1)
            parser_class = getattr(importlib.import_module(module_name), class_name)

        if not issubclass(parser_class, Parser):
            raise ValueError("specified parser does not inherit audeep.backend.parsers.Parser")

        parser = parser_class(parsed_args.basedir)

        if not parser.can_parse():
            raise ValueError("specified parser is unable to parse data set at {}".format(parsed_args.basedir))

        lengths = []
        sample_rates = []
        channels = []

        non_seekable_files = False

        instance_metadata = parser.parse()

        self.log.info("reading audio file information")

        for index, metadata in enumerate(instance_metadata):
            self.log.debug("processing %%s (%%%dd/%%d)" % int(math.ceil(math.log10(len(instance_metadata)))),
                           metadata.path, index + 1, len(instance_metadata))

            with SoundFile(str(metadata.path)) as sf:
                sample_rates.append(sf.samplerate)
                channels.append(sf.channels)

                if sf.seekable():
                    lengths.append(sf.seek(0, SEEK_END) / sf.samplerate)
                else:
                    non_seekable_files = True

        if non_seekable_files:
            self.log.warning("could not determine the length of some files - information may be inaccurate")

        information = [
            ("number of audio files", parser.num_instances),
            ("number of labels", len(parser.label_map) if parser.label_map else "0"),
            ("cross validation folds", parser.num_folds),
            ("minimum sample length", "%.2f s" % np.min(lengths)),
            ("maximum sample length", "%.2f s" % np.max(lengths)),
        ]

        if len(np.unique(sample_rates)) > 1:
            information.append(("sample rates", "%d Hz - %d Hz" % (np.min(sample_rates), np.max(sample_rates))))
        else:
            information.append(("sample rate", "%d Hz" % sample_rates[0]))

        if len(np.unique(channels)) > 1:
            information.append(("channels", "%d - %d" % (np.min(channels), np.max(channels))))
        else:
            information.append(("channels", channels[0]))

        TableFormatter("lr").print(data=information,
                                   header="data set information")


class InspectNetCDF(Command):
    """
    Display information about a data set in netCDF4 format
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep data "
                                 "model")
        parser.add_argument("--instance",
                            metavar="INDEX",
                            default=None,
                            type=int,
                            help="Additionally display information about the instance at the specified index")
        parser.add_argument("--detailed-folds",
                            action="store_true",
                            help="Show detailed information about cross-validation folds, if present")

        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("unable to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)
        formatter = TableFormatter()

        # print global information
        global_information = [
            ("number of instances", data_set.num_instances),
            ("cross validation info", data_set.has_cv_info),
            ("partition info", data_set.has_partition_info),
            ("fully labeled", data_set.is_fully_labeled),
            ("feature dimensions", data_set.feature_dims),
        ]

        print()
        formatter.print(data=global_information,
                        header="global data set information")
        print()

        # print instance information
        if parsed_args.instance is not None:
            instance = data_set[parsed_args.instance]

            instance_information = [
                ("data file", instance.filename),
                ("chunk number", instance.chunk_nr),
                ("label", "{} ({})".format(instance.label_nominal, instance.label_numeric)),
                ("cross validation splits",
                 ", ".join(["None" if x is None else x.name for x in instance.cv_folds]) or None),
                ("partition", None if instance.partition is None else instance.partition.name),
                ("shape", instance.feature_shape),
            ]

            formatter.print(data=instance_information,
                            header="instance {} information:".format(parsed_args.instance))
            print()

        if parsed_args.detailed_folds and data_set.has_cv_info and data_set.is_fully_labeled:
            formatter = TableFormatter(alignment="lrrrr")

            inverse_label_map = dict(map(reversed, data_set.label_map.items()))

            for fold in range(data_set.num_folds):
                train_split = data_set.split(fold, Split.TRAIN)
                valid_split = data_set.split(fold, Split.VALID)

                labels, train_counts = np.unique(train_split.labels_numeric, return_counts=True)
                _, valid_counts = np.unique(valid_split.labels_numeric, return_counts=True)

                train_total = sum(train_counts)
                valid_total = sum(valid_counts)

                fold_information = [
                ]

                for i in range(len(labels)):
                    train_count = train_counts[i]
                    valid_count = valid_counts[i]

                    train_relative = 100 * train_count / train_total
                    valid_relative = 100 * valid_count / valid_total

                    fold_information.append((inverse_label_map[labels[i]],
                                             train_count, "%2.2f%%" % train_relative,
                                             valid_count, "%2.2f%%" % valid_relative))

                fold_information.append(("total", train_total, "", valid_total, ""))

                formatter.print(data=fold_information,
                                header="fold {} information:".format(fold + 1),
                                dividers=[len(labels) - 1])
                print()
