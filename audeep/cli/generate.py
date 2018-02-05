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

"""Feature generation commands"""
import abc
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load
from audeep.backend.training.base import BaseFeatureLearningWrapper
from audeep.backend.training.frequency_autoencoder import FrequencyAutoencoderWrapper
from audeep.backend.training.frequency_time_autoencoder import FrequencyTimeAutoencoderWrapper
from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper


class GenerateBaseCommand(Command):
    """
    Base class for all feature generation commands.
    
    Defines common command line options and common functionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 app,
                 app_args,
                 wrapper: BaseFeatureLearningWrapper,
                 default_batch_size: int = 500):
        """
        Creates and initializes a new GenerateBaseCommand with the specified parameters.
        
        Parameters
        ----------
        app
            Pass through to `Command`
        app_args
            Pass through to `Command`
        wrapper: FeatureLearningWrapper
            The feature learning wrapper used for generating features
        default_batch_size: int
            Default batch size
        """
        super().__init__(app, app_args)

        self._wrapper = wrapper
        self._default_batch_size = default_batch_size

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--batch-size",
                            default=self._default_batch_size,
                            type=int,
                            help="The number of examples per minibatch (default %d)" % self._default_batch_size)
        parser.add_argument("--model-dir",
                            type=Path,
                            required=True,
                            help="Directory containing model files. Typically, this is the 'logs' subdirectory of a"
                                 "run directory (--run-name option of the 'rae train' command).")
        parser.add_argument("--steps",
                            nargs="+",
                            type=int,
                            default=None,
                            help="Use models at the specified global steps. Defaults to the latest step.")
        parser.add_argument("--output",
                            nargs="+",
                            type=Path,
                            required=True,
                            help="Files to which to write the generated features (one for each model). Data is stored "
                                 "in netCDF 4 format according to the auDeep data model.")
        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep data "
                                 "model")

        return parser

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        steps = parsed_args.steps

        if steps is None:
            steps = [None]

        if len(steps) != len(parsed_args.output):
            raise ValueError("There must be one output file for each global step")

        input_data = load(parsed_args.input)

        for step, output_file in zip(steps, parsed_args.output):
            generated_data = self._wrapper.generate(model_filename=parsed_args.model_dir / "model",
                                                    global_step=step,
                                                    data_set=input_data,
                                                    batch_size=parsed_args.batch_size)
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True)

            generated_data.save(output_file)


class GenerateTimeAutoencoder(GenerateBaseCommand):
    """
    Generate features using a trained frequency recurrent autoencoder
    """

    def __init__(self,
                 app,
                 app_args):
        super().__init__(app, app_args, TimeAutoencoderWrapper())


class GenerateFrequencyAutoencoder(GenerateBaseCommand):
    """
    Generate features using a trained frequency recurrent autoencoder
    """

    def __init__(self,
                 app,
                 app_args):
        super().__init__(app, app_args,
                         FrequencyAutoencoderWrapper(),
                         default_batch_size=64)


class GenerateFrequencyTimeAutoencoder(GenerateBaseCommand):
    """
    Generate features using a trained frequency-time recurrent autoencoder
    """

    def __init__(self,
                 app,
                 app_args):
        super().__init__(app, app_args, FrequencyTimeAutoencoderWrapper())
