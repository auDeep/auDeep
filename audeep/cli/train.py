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

"""DNN training commands"""
import abc
import shutil
import tempfile
from pathlib import Path

from cliff.command import Command

from audeep.backend.data.data_set import load
from audeep.backend.data.export import export_tfrecords
from audeep.backend.enum_parser import EnumType
from audeep.backend.log import LoggingMixin
from audeep.backend.models.rnn_base import CellType, RNNArchitecture
from audeep.backend.training.frequency_autoencoder import FrequencyAutoencoderWrapper
from audeep.backend.training.frequency_time_autoencoder import FrequencyTimeAutoencoderWrapper
from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper


class TrainBaseCommand(LoggingMixin, Command):
    """
    Base class for all training commands.
    
    Defines common command line options and common functionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 app,
                 app_args,
                 default_batch_size: int = 64,
                 default_num_epochs: int = 10,
                 default_learning_rate: float = 0.001,
                 default_run_name: Path = Path("./test-run")):
        """
        Create and initialize a new TrainBaseCommand with the specified parameters.
        
        Parameters
        ----------
        app
            Pass through to `Command`
        app_args
            Pass through to `Command`
        default_batch_size: int
            Default batch size
        default_num_epochs: int
            Default number of epochs
        default_learning_rate: float
            Default learning rate
        default_run_name: Path
            Default run name
        """
        super().__init__(app, app_args)

        self.default_batch_size = default_batch_size
        self.default_num_epochs = default_num_epochs
        self.default_learning_rate = default_learning_rate
        self.default_run_name = default_run_name

        self.model_filename = None
        self.record_files = None
        self.feature_shape = None
        self.num_instances = None

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--batch-size",
                            default=self.default_batch_size,
                            type=int,
                            help="The minibatch size (default %d)" % self.default_batch_size)
        parser.add_argument("--num-epochs",
                            default=self.default_num_epochs,
                            type=int,
                            help="The number of training epochs (default %d)" % self.default_num_epochs)
        parser.add_argument("--learning-rate",
                            default=self.default_learning_rate,
                            type=float,
                            help="The learning rate (default %1.1e)" % self.default_learning_rate)
        parser.add_argument("--run-name",
                            default=self.default_run_name,
                            type=Path,
                            help="Base directory for the training run (default %s)" % self.default_run_name)
        parser.add_argument("--checkpoints-to-keep",
                            default=None,
                            type=int,
                            help="Number of checkpoints to keep (default all). If set, only the most recent checkpoints"
                                 "will be kept.")
        parser.add_argument("--input",
                            type=Path,
                            nargs="+",
                            required=True,
                            help="Files containing data sets in netCDF 4 format, conformant to the auDeep data "
                                 "model")
        parser.add_argument("--tempdir",
                            type=Path,
                            default=None,
                            help="A directory for temporary files. Defaults to the OS temp location. The entire "
                                 "training data is written to this directory in TFRecords format, so make sure there is"
                                 "enough disk space available.")
        parser.add_argument("--continue",
                            dest="continue_training",
                            action="store_true",
                            help="Continue training from the latest checkpoint. Ignores all parameters concerning "
                                 "network architecture.")

        return parser

    def _setup_io(self,
                  parsed_args,
                  tempdir: Path):
        data_files = parsed_args.input

        for file in data_files:
            if not file.exists():
                raise IOError("failed to open data set file at {}".format(file))

        self.record_files = []
        self.num_instances = 0

        # convert data sets to tfrecords and collect metadata
        for index, file in enumerate(data_files):
            record_file = tempdir / (file.name + ("-%d" % index))
            self.record_files.append(record_file)

            self.log.info("created temporary file %s for data set %s", record_file, file)

            data_set = load(file)

            if self.feature_shape is None:
                self.feature_shape = data_set.feature_shape
            elif self.feature_shape != data_set.feature_shape:
                raise ValueError("data sets have different feature shapes")

            self.num_instances += data_set.num_instances

            export_tfrecords(record_file, data_set)

        # create output dirs
        output_dir = parsed_args.run_name

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        self.model_filename = output_dir / "logs" / "model"

        if not self.model_filename.parent.exists():
            self.model_filename.parent.mkdir()

    def run(self, parsed_args):
        if parsed_args.tempdir is None:
            tempdir = Path(tempfile.mkdtemp())
        else:
            tempdir = parsed_args.tempdir
            tempdir.mkdir(parents=True)

        self._setup_io(parsed_args, tempdir)

        if parsed_args.continue_training and not self.model_filename.with_suffix(".meta").exists():
            self.log.error("The --continue option is set but no previous metagraph was found at %s. Re-run the command "
                           "without the --continue option to start a new training run.",
                           self.model_filename.with_suffix(".meta"))
            return 1
        elif not parsed_args.continue_training and self.model_filename.with_suffix(".meta").exists():
            self.log.error("A previous metagraph was found at %s. Use the --continue option to continue training from "
                           "the previous checkpoint, or change the run name to a different location.",
                           self.model_filename.with_suffix(".meta"))
            return 1

        try:
            retval = super().run(parsed_args)
        except Exception as e:
            raise
        finally:
            self.log.debug("removing temporary directory %s", tempdir)
            shutil.rmtree(str(tempdir), ignore_errors=True)

        return retval or 1

    @abc.abstractmethod
    def take_action(self, parsed_args):
        pass


class TrainAutoencoderBaseCommand(TrainBaseCommand):
    """
    Base command for autoencoder training commands.
    
    Defines common command line options, and common functionality.
    """
    __metaclass__ = abc.ABCMeta

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--num-layers",
                            default=1,
                            type=int,
                            help="The number of layers in the encoder and decoder (default 1)")
        parser.add_argument("--num-units",
                            default=16,
                            type=int,
                            help="The number of RNN cells per layer (default 16)")
        parser.add_argument("--bidirectional-encoder",
                            action="store_true",
                            help="Use a bidirectional encoder (default off)")
        parser.add_argument("--bidirectional-decoder",
                            action="store_true",
                            help="Use a bidirectional decoder (default off)")
        parser.add_argument("--cell",
                            default=CellType.GRU,
                            type=EnumType(CellType),
                            help="The type of the RNN cells (GRU or LSTM, default GRU)")
        parser.add_argument("--keep-prob",
                            default=0.8,
                            type=float,
                            help="Keep activations with the specified probability (default 0.8)")

        return parser

    @abc.abstractmethod
    def take_action(self, parsed_args):
        pass


class TrainTimeAutoencoder(TrainAutoencoderBaseCommand):
    """
    Train a time-recurrent autoencoder on spectrograms
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--encoder-noise",
                            default=0.0,
                            type=float,
                            help="Replace encoder input time steps by zeros with the specified probability "
                                 "(default 0.0)")
        parser.add_argument("--feed-previous-prob",
                            default=0.0,
                            type=float,
                            help="Feed output of previous time step instead of correct output to decoder with "
                                 "specified probability (default 0.0)")
        parser.add_argument("--mask-silence",
                            action="store_true",
                            help="Mask silence in the loss function (experimental)")

        return parser

    def take_action(self, parsed_args):
        encoder_architecture = RNNArchitecture(num_layers=parsed_args.num_layers,
                                               num_units=parsed_args.num_units,
                                               bidirectional=parsed_args.bidirectional_encoder,
                                               cell_type=parsed_args.cell)
        decoder_architecture = RNNArchitecture(num_layers=parsed_args.num_layers,
                                               num_units=parsed_args.num_units,
                                               bidirectional=parsed_args.bidirectional_decoder,
                                               cell_type=parsed_args.cell)

        wrapper = TimeAutoencoderWrapper()

        if not parsed_args.continue_training:
            wrapper.initialize_model(feature_shape=self.feature_shape,
                                     model_filename=self.model_filename,
                                     encoder_architecture=encoder_architecture,
                                     decoder_architecture=decoder_architecture,
                                     mask_silence=parsed_args.mask_silence)

        wrapper.train_model(model_filename=self.model_filename,
                            record_files=self.record_files,
                            feature_shape=self.feature_shape,
                            num_instances=self.num_instances,
                            num_epochs=parsed_args.num_epochs,
                            batch_size=parsed_args.batch_size,
                            checkpoints_to_keep=parsed_args.checkpoints_to_keep,
                            learning_rate=parsed_args.learning_rate,
                            keep_prob=parsed_args.keep_prob,
                            encoder_noise=parsed_args.encoder_noise,
                            decoder_feed_previous_prob=parsed_args.feed_previous_prob)


class TrainFrequencyAutoencoder(TrainAutoencoderBaseCommand):
    """
    Train a frequency-recurrent autoencoder on spectrograms
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--freq-window-width",
                            default=32,
                            type=int,
                            help="the width of the sliding window on the frequency axis (default 32)")
        parser.add_argument("--freq-window-overlap",
                            default=24,
                            type=int,
                            help="overlap between windows on the frequency axis (default 24)")
        parser.add_argument("--encoder-noise",
                            default=0.0,
                            type=float,
                            help="Replace encoder input time steps by zeros with the specified probability "
                                 "(default 0.0)")
        parser.add_argument("--feed-previous-prob",
                            default=0.0,
                            type=float,
                            help="Feed output of previous time step instead of correct output to decoder with "
                                 "specified probability (default 0.0)")
        return parser

    def take_action(self, parsed_args):
        encoder_architecture = RNNArchitecture(num_layers=parsed_args.num_layers,
                                               num_units=parsed_args.num_units,
                                               bidirectional=parsed_args.bidirectional_encoder,
                                               cell_type=parsed_args.cell)
        decoder_architecture = RNNArchitecture(num_layers=parsed_args.num_layers,
                                               num_units=parsed_args.num_units,
                                               bidirectional=parsed_args.bidirectional_decoder,
                                               cell_type=parsed_args.cell)

        wrapper = FrequencyAutoencoderWrapper()

        if not parsed_args.continue_training:
            wrapper.initialize_model(feature_shape=self.feature_shape,
                                     model_filename=self.model_filename,
                                     encoder_architecture=encoder_architecture,
                                     decoder_architecture=decoder_architecture,
                                     frequency_window_width=parsed_args.freq_window_width,
                                     frequency_window_overlap=parsed_args.freq_window_overlap)

        wrapper.train_model(model_filename=self.model_filename,
                            record_files=self.record_files,
                            feature_shape=self.feature_shape,
                            num_instances=self.num_instances,
                            num_epochs=parsed_args.num_epochs,
                            batch_size=parsed_args.batch_size,
                            checkpoints_to_keep=parsed_args.checkpoints_to_keep,
                            learning_rate=parsed_args.learning_rate,
                            keep_prob=parsed_args.keep_prob,
                            encoder_noise=parsed_args.encoder_noise,
                            decoder_feed_previous_prob=parsed_args.feed_previous_prob)


class TrainFrequencyTimeAutoencoder(TrainBaseCommand):
    """
    Train a frequency-time-recurrent autoencoder on spectrograms
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--num-f-layers",
                            default=1,
                            type=int,
                            help="The number of layers in the frequency encoder and decoder (default 1)")
        parser.add_argument("--num-f-units",
                            default=64,
                            type=int,
                            help="The number of RNN cells per layer in the frequency RNNs(default 64)")
        parser.add_argument("--num-t-layers",
                            default=2,
                            type=int,
                            help="The number of layers in the time encoder and decoder (default 2)")
        parser.add_argument("--num-t-units",
                            default=128,
                            type=int,
                            help="The number of RNN cells per layer in the time RNNs (default 128)")
        parser.add_argument("--bidirectional-f-encoder",
                            action="store_true",
                            help="Use a bidirectional frequency encoder (default off)")
        parser.add_argument("--bidirectional-f-decoder",
                            action="store_true",
                            help="Use a bidirectional frequency decoder (default off)")
        parser.add_argument("--bidirectional-t-encoder",
                            action="store_true",
                            help="Use a bidirectional time encoder (default off)")
        parser.add_argument("--bidirectional-t-decoder",
                            action="store_true",
                            help="Use a bidirectional time decoder (default off)")
        parser.add_argument("--cell",
                            default=CellType.GRU,
                            type=EnumType(CellType),
                            help="The type of the RNN cells (GRU or LSTM, default GRU)")
        parser.add_argument("--keep-prob",
                            default=0.8,
                            type=float,
                            help="Keep activations with the specified probability (default 0.8)")
        parser.add_argument("--freq-window-width",
                            default=32,
                            type=int,
                            help="the width of the sliding window on the frequency axis (default 32)")
        parser.add_argument("--freq-window-overlap",
                            default=24,
                            type=int,
                            help="overlap between windows on the frequency axis (default 24)")
        parser.add_argument("--f-encoder-noise",
                            default=0.0,
                            type=float,
                            help="Replace frequency encoder input time steps by zeros with the specified probability "
                                 "(default 0.0)")
        parser.add_argument("--t-encoder-noise",
                            default=0.0,
                            type=float,
                            help="Replace time encoder input time steps by zeros with the specified probability "
                                 "(default 0.0)")
        parser.add_argument("--f-feed-previous-prob",
                            default=0.0,
                            type=float,
                            help="Feed output of previous time step instead of correct output to frequency decoder "
                                 "with specified probability (default 0.0)")
        parser.add_argument("--t-feed-previous-prob",
                            default=0.0,
                            type=float,
                            help="Feed output of previous time step instead of correct output to time decoder "
                                 "with specified probability (default 0.0)")

        return parser

    def take_action(self, parsed_args):
        f_encoder_architecture = RNNArchitecture(num_layers=parsed_args.num_f_layers,
                                                 num_units=parsed_args.num_f_units,
                                                 bidirectional=parsed_args.bidirectional_f_encoder,
                                                 cell_type=parsed_args.cell)
        t_encoder_architecture = RNNArchitecture(num_layers=parsed_args.num_t_layers,
                                                 num_units=parsed_args.num_t_units,
                                                 bidirectional=parsed_args.bidirectional_t_encoder,
                                                 cell_type=parsed_args.cell)
        f_decoder_architecture = RNNArchitecture(num_layers=parsed_args.num_f_layers,
                                                 num_units=parsed_args.num_f_units,
                                                 bidirectional=parsed_args.bidirectional_f_decoder,
                                                 cell_type=parsed_args.cell)
        t_decoder_architecture = RNNArchitecture(num_layers=parsed_args.num_t_layers,
                                                 num_units=parsed_args.num_t_units,
                                                 bidirectional=parsed_args.bidirectional_t_decoder,
                                                 cell_type=parsed_args.cell)

        wrapper = FrequencyTimeAutoencoderWrapper()

        if not parsed_args.continue_training:
            wrapper.initialize_model(feature_shape=self.feature_shape,
                                     model_filename=self.model_filename,
                                     f_encoder_architecture=f_encoder_architecture,
                                     t_encoder_architecture=t_encoder_architecture,
                                     f_decoder_architecture=f_decoder_architecture,
                                     t_decoder_architecture=t_decoder_architecture,
                                     frequency_window_width=parsed_args.freq_window_width,
                                     frequency_window_overlap=parsed_args.freq_window_overlap)

        wrapper.train_model(model_filename=self.model_filename,
                            record_files=self.record_files,
                            feature_shape=self.feature_shape,
                            num_instances=self.num_instances,
                            num_epochs=parsed_args.num_epochs,
                            batch_size=parsed_args.batch_size,
                            checkpoints_to_keep=parsed_args.checkpoints_to_keep,
                            learning_rate=parsed_args.learning_rate,
                            keep_prob=parsed_args.keep_prob,
                            f_encoder_noise=parsed_args.f_encoder_noise,
                            t_encoder_noise=parsed_args.t_encoder_noise,
                            f_decoder_feed_previous_prob=parsed_args.f_feed_previous_prob,
                            t_decoder_feed_previous_prob=parsed_args.t_feed_previous_prob)
