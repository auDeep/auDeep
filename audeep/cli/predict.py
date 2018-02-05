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

"""Commands for generating predictions on test data"""
import abc
from pathlib import Path

import pandas as pd
from cliff.command import Command

from audeep.backend.data.data_set import Partition, load
from audeep.backend.enum_parser import EnumType
from audeep.backend.learners import LearnerBase, PreProcessingWrapper, LibLINEARLearner, TensorflowMLPLearner
from audeep.backend.log import LoggingMixin


class PredictBaseCommand(LoggingMixin, Command):
    """
    Base class for all prediction commands.
    
    This class defines common command line options, and common functionality.
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

        self._learner = None

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--train-input",
                            type=Path,
                            required=True,
                            help="File containing training data in netCDF 4 format, conformant to the auDeep"
                                 "data model")
        parser.add_argument("--train-partitions",
                            nargs="+",
                            type=EnumType(Partition),
                            help="Use only the specified partitions of the training data (TRAIN, DEVEL, or TEST)")
        parser.add_argument("--eval-input",
                            type=Path,
                            required=True,
                            help="File containing evaluation data in netCDF 4 format, conformant to the auDeep"
                                 "data model")
        parser.add_argument("--eval-partitions",
                            nargs="+",
                            type=EnumType(Partition),
                            help="Use only the specified partitions of the evaluation data (TRAIN, DEVEL, or TEST)")
        parser.add_argument("--upsample",
                            action="store_true",
                            help="Balance classes in the training data")
        parser.add_argument("--majority-vote",
                            action="store_true",
                            help="Use majority voting to determine the labels of chunked instances")
        parser.add_argument("--output",
                            type=Path,
                            required=True,
                            help="Write predictions to the specified file as CSV with tab delimiters")

        return parser

    @abc.abstractmethod
    def _get_learner(self, parsed_args) -> LearnerBase:
        pass

    def take_action(self, parsed_args):
        self._learner = PreProcessingWrapper(learner=self._get_learner(parsed_args),
                                             upsample=parsed_args.upsample,
                                             majority_vote=parsed_args.majority_vote)

        if not parsed_args.train_input.exists():
            raise IOError("failed to open training data file at {}".format(parsed_args.train_input))
        if not parsed_args.eval_input.exists():
            raise IOError("failed to open evaluation data file at {}".format(parsed_args.eval_input))

        train_data = load(parsed_args.train_input)
        eval_data = load(parsed_args.eval_input)

        if parsed_args.train_partitions is not None:
            train_data = train_data.partitions(parsed_args.train_partitions)

        if parsed_args.eval_partitions is not None:
            eval_data = eval_data.partitions(parsed_args.eval_partitions)

        self.log.info("training classifier")

        self._learner.fit(train_data)
        predictions = self._learner.predict(eval_data)

        inverse_label_map = dict(map(reversed, train_data.label_map.items()))
        predictions = [(item[0], inverse_label_map[item[1]]) for item in
                       sorted(predictions.items(), key=lambda item: item[0])]

        self.log.info("writing predictions to %s", parsed_args.output)

        if not parsed_args.output.parent.exists():
            parsed_args.output.parent.mkdir(parents=True)

        output = pd.DataFrame.from_records(predictions, index=range(len(predictions)))
        output.to_csv(parsed_args.output, sep="\t", index=False, header=False)


class MLPPrediction(PredictBaseCommand):
    """
    Generate predictions on some data using an MLP
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--learning-rate",
                            type=float,
                            default=0.001,
                            help="The learning rate (default 0.001)")
        parser.add_argument("--num-epochs",
                            type=int,
                            default=400,
                            help="The number of training epochs (default 400)")
        parser.add_argument("--num-layers",
                            type=int,
                            default=2,
                            help="The number of hidden layers (default 2)")
        parser.add_argument("--num-units",
                            type=int,
                            default=150,
                            help="The number of nodes per hidden layer (default 150)")
        parser.add_argument("--keep-prob",
                            type=float,
                            default=0.6,
                            help="The probability to keep hidden activations during training (default 0.6)")
        parser.add_argument("--shuffle",
                            action="store_true",
                            help="Shuffle training data in each epoch (default off)")

        return parser

    def _get_learner(self, parsed_args) -> LearnerBase:
        checkpoint_dir = parsed_args.train_input.parent / "evaluation"  # type: Path

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        return TensorflowMLPLearner(checkpoint_dir=checkpoint_dir,
                                    num_layers=parsed_args.num_layers,
                                    num_units=parsed_args.num_units,
                                    learning_rate=parsed_args.learning_rate,
                                    num_epochs=parsed_args.num_epochs,
                                    keep_prob=parsed_args.keep_prob,
                                    shuffle_training=parsed_args.shuffle)


class SVMPrediction(PredictBaseCommand):
    """
    Generate predictions on some data using LibLINEAR
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--complexity",
                            type=float,
                            required=True,
                            help="The SVM complexity")

        return parser

    def _get_learner(self, parsed_args) -> LearnerBase:
        return LibLINEARLearner(complexity=parsed_args.complexity)
