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

"""Spectrogram extraction command"""
import importlib
import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from cliff.command import Command

from audeep.backend.data.data_set import empty
from audeep.backend.enum_parser import EnumType
from audeep.backend.formatters import TableFormatter
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata
from audeep.backend.preprocessing import Preprocessor, ChannelFusion


class ExtractSpectrograms(LoggingMixin, Command):
    """
    Parses a data set and extracts spectrograms from audio files
    """

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
        parser.add_argument("--output",
                            type=Path,
                            default=Path("./spectrograms.nc"),
                            help="The output filename (default './spectrograms.nc'). Data is stored in netCDF 4 format"
                                 "according to the auDeep data model.")
        parser.add_argument("--mel-spectrum",
                            default=None,
                            type=int,
                            metavar="FILTERS",
                            help="Generate mel spectrograms with the specified number of filter banks (default off)")
        parser.add_argument("--window-width",
                            default=0.04,
                            type=float,
                            help="The width of the FFT window in seconds (default 0.04)")
        parser.add_argument("--window-overlap",
                            default=0.02,
                            type=float,
                            help="The overlap between FFT windows in seconds (default 0.02)")
        parser.add_argument("--clip-above",
                            metavar="dB",
                            default=None,
                            type=int,
                            help="Clip amplitudes above the specified dB value. Amplitudes are normalized so that 0dB "
                                 "is the highest value.")
        parser.add_argument("--clip-below",
                            metavar="dB",
                            default=None,
                            type=int,
                            help="Clip amplitudes below the specified dB value. Amplitudes are normalized so that 0dB "
                                 "is the highest value.")
        parser.add_argument("--chunk-length",
                            default=None,
                            type=float,
                            help="Split audio files into chunks of specified length in seconds. Requires the "
                                 "--chunk-count option to be set.")
        parser.add_argument("--chunk-count",
                            default=None,
                            type=int,
                            help="Number of chunks per audio file. Excess chunks will be dropped, and an error is "
                                 "raised if there are not enough chunks. Requires the --chunk-length option to be set.")
        parser.add_argument("--pretend",
                            default=None,
                            type=int,
                            help="Process the file at the specified index only and display the resulting spectrogram")
        parser.add_argument("--channels",
                            default=ChannelFusion.MEAN,
                            type=EnumType(ChannelFusion),
                            help="Strategy for combining the audio channels. Valid values are \"mean\", \"left\", "
                                 "\"right\", and \"diff\".")
        parser.add_argument("--fixed-length",
                            default=None,
                            type=float,
                            help="Ensure that all samples have exactly the specified length in seconds, by cutting or "
                                 "padding audio appropriately")
        parser.add_argument("--center-fixed",
                            action="store_true",
                            help="Pad or cut equally at the start and end of samples if a fixed length is set. By "
                                 "default, padding or cutting is performed only at the end of samples.")

        return parser

    def get_preprocessor(self, parsed_args):
        return Preprocessor(channel_fusion=parsed_args.channels,
                            pre_emphasis=True,
                            window_width=parsed_args.window_width,
                            window_overlap=parsed_args.window_overlap,
                            mel_bands=parsed_args.mel_spectrum,
                            clip_power_above=parsed_args.clip_above,
                            clip_power_below=parsed_args.clip_below,
                            chunk_length=parsed_args.chunk_length,
                            chunk_count=parsed_args.chunk_count,
                            fixed_length=parsed_args.fixed_length,
                            center_fixed=parsed_args.center_fixed,
                            mean_norm=parsed_args.pretend is None)

    def take_action(self, parsed_args):
        if (parsed_args.chunk_count is not None) ^ (parsed_args.chunk_length is not None):
            raise ValueError("--chunk-count can only be used with --chunk-length and vice-versa")

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
        preprocessor = self.get_preprocessor(parsed_args)

        if not parser.can_parse():
            raise ValueError("specified parser is unable to parse data set at {}".format(parsed_args.basedir))

        self.log.info("parsing data set at %s", parsed_args.basedir)

        instance_metadata = parser.parse()

        if parsed_args.pretend is not None:
            metadata = instance_metadata[parsed_args.pretend]  # type: _InstanceMetadata

            sxx, f, t = preprocessor.process(metadata.path)

            # noinspection PyTypeChecker
            spectrogram_info = [("data file", metadata.path)]

            if parser.label_map is not None:
                spectrogram_info.append(("label", "{} ({})"
                                         .format(metadata.label_nominal, parser.label_map[metadata.label_nominal])))
            else:
                spectrogram_info.append(("label", "{} ({})"
                                         .format(metadata.label_nominal, metadata.label_numeric)))

            if metadata.cv_folds:
                # noinspection PyTypeChecker
                spectrogram_info.append(("cross validation splits", ",".join([x.name for x in metadata.cv_folds])))

            if metadata.partition is not None:
                spectrogram_info.append(("partition", metadata.partition.name))

            # noinspection PyTypeChecker
            spectrogram_info.append(("number of chunks", len(sxx)))
            spectrogram_info.append(("spectrogram time steps", [x.shape[1] for x in sxx]))
            spectrogram_info.append(("spectrogram frequency bands", f.shape[0]))

            TableFormatter().print(data=spectrogram_info)

            fig = plt.figure()
            sxx_full = np.concatenate(sxx, axis=1)
            t_full = np.concatenate(t)

            nxticks = sxx_full.shape[1] // 25
            nyticks = 4

            # spectrogram
            ax = fig.add_subplot(2, 1, 1)
            plt.title("Spectrogram")
            ax.set_xticks(np.arange(0, t_full.shape[0], t_full.shape[0] // nxticks))
            ax.set_xticklabels(np.round(t_full[::t_full.shape[0] // nxticks]))
            ax.set_xlabel("Time (s)")

            ax.set_yticks(np.arange(0, len(f), len(f) // nyticks))
            ax.set_yticklabels(np.round(f[::-len(f) // nyticks]))
            ax.set_ylabel("Frequency (Hz)")

            ax.imshow(sxx_full[::-1], cmap="magma")

            # histogram
            ax = fig.add_subplot(2, 1, 2)
            plt.title("Amplitude Histogram")
            ax.set_xlabel("Amplitude (dB)")
            ax.set_ylabel("Probability")

            range_min = parsed_args.clip_below + 0.01 if parsed_args.clip_below is not None else sxx_full.min()
            range_max = parsed_args.clip_above - 0.01 if parsed_args.clip_above is not None else 0

            ax.hist(sxx_full.flatten(),
                    range=(range_min, range_max),
                    bins=100,
                    density=True,
                    histtype="stepfilled")

            plt.tight_layout()
            plt.show()
        else:
            num_instances = parser.num_instances * (1 if parsed_args.chunk_count is None else parsed_args.chunk_count)
            data_set = None

            index = 0

            for file_index, metadata in enumerate(instance_metadata):  # type: Tuple[int, _InstanceMetadata]
                self.log.info("processing %%s (%%%dd/%%d)" % int(math.ceil(math.log10(len(instance_metadata)))),
                              metadata.path, file_index + 1, len(instance_metadata))

                sxx, _, _ = preprocessor.process(metadata.path)

                for chunk_nr, sxx_chunk in enumerate(sxx):
                    if data_set is None:
                        data_set = empty(num_instances=num_instances,
                                         feature_dimensions=[("time", sxx_chunk.shape[1]),
                                                             ("freq", sxx_chunk.shape[0])],
                                         num_folds=parser.num_folds)
                        data_set.label_map = parser.label_map

                    instance = data_set[index]
                    instance.filename = metadata.filename
                    instance.chunk_nr = chunk_nr
                    instance.label_nominal = metadata.label_nominal

                    if data_set.label_map is None:
                        instance.label_numeric = metadata.label_numeric

                    instance.cv_folds = metadata.cv_folds
                    instance.partition = metadata.partition
                    instance.features = np.transpose(sxx_chunk)

                    index += 1

            data_set.save(parsed_args.output)
