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

"""Evaluation commands"""
import abc
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cliff.command import Command

from audeep.backend.data.data_set import load, Partition
from audeep.backend.enum_parser import EnumType
from audeep.backend.evaluation import CrossValidatedEvaluation, PartitionedEvaluation
from audeep.backend.formatters import ConfusionMatrixFormatter
from audeep.backend.learners import TensorflowMLPLearner, LearnerBase, LibLINEARLearner
from audeep.backend.log import LoggingMixin


class EvaluateBaseCommand(LoggingMixin, Command):
    """
    Base class for all evaluation commands.
    
    This class defines common command line options, and common functionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 app,
                 app_args):
        super().__init__(app, app_args)

        self._learner = None

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing generated features in netCDF 4 format, conformant to the auDeep"
                                 "data model")
        parser.add_argument("--cross-validate",
                            action="store_true",
                            help="Use cross-validation according to the cross-validation setup of the data")
        parser.add_argument("--train-partitions",
                            nargs="+",
                            type=EnumType(Partition),
                            help="Train classifier on the specified partitions (TRAIN, DEVEL, or TEST)")
        parser.add_argument("--eval-partitions",
                            nargs="+",
                            type=EnumType(Partition),
                            help="Evaluate classifier on the specified partitions (TRAIN, DEVEL, or TEST)")
        parser.add_argument("--upsample",
                            action="store_true",
                            help="Balance classes in the training partitions/splits")
        parser.add_argument("--majority-vote",
                            action="store_true",
                            help="Use majority voting to determine the labels of chunked instances")
        parser.add_argument("--repeat",
                            metavar="N",
                            default=1,
                            type=int,
                            help="Repeat evaluation N times and compute the mean accuracy (default 1)")

        return parser

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap="Blues"):
        """
        This function plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm_norm

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "%1.2f" % cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "%.0f" % cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @abc.abstractmethod
    def _get_learner(self, parsed_args) -> LearnerBase:
        pass

    def take_action(self, parsed_args):
        self._learner = self._get_learner(parsed_args)

        if not parsed_args.input.exists():
            raise IOError("failed to open data file at {}".format(parsed_args.input))

        if parsed_args.train_partitions is None and parsed_args.eval_partitions is None and not parsed_args.cross_validate:
            raise ValueError("must select either cross validated or partitioned evaluation")
        if (parsed_args.train_partitions is not None or parsed_args.eval_partitions is not None) \
                and parsed_args.cross_validate:
            raise ValueError("partitioned evaluation and cross validated evaluation are mutually exclusive")
        if not parsed_args.cross_validate and ((parsed_args.train_partitions is not None)
                                                   ^ (parsed_args.eval_partitions is not None)):
            raise ValueError("at least one train and eval partition required")

        data_set = load(parsed_args.input)

        accuracies = []
        uars = []
        confusion_matrices = []

        if parsed_args.cross_validate:
            accuracy_confidence_intervals = []
            uar_confidence_intervals = []

            for _ in range(parsed_args.repeat):
                evaluation = CrossValidatedEvaluation(learner=self._learner,
                                                      upsample=parsed_args.upsample,
                                                      majority_vote=parsed_args.majority_vote)
                evaluation.run(data_set)

                accuracies.append(evaluation.accuracy)
                uars.append(evaluation.uar)
                accuracy_confidence_intervals.append(evaluation.accuracy_confidence_interval)
                uar_confidence_intervals.append(evaluation.uar_confidence_interval)
                confusion_matrices.append(evaluation.confusion_matrix)

            accuracy = np.mean(accuracies)
            accuracy_confidence_interval = np.mean(accuracy_confidence_intervals)
            uar = np.mean(uars)
            uar_confidence_interval = np.mean(uar_confidence_intervals)

            self.log.info("cross validation accuracy: %2.2f%% (+/- %2.2f%%)", 100 * accuracy,
                          100 * accuracy_confidence_interval)
            self.log.info("cross validation UAR: %2.2f%% (+/- %2.2f%%)", 100 * uar, 100 * uar_confidence_interval)
        else:
            for _ in range(parsed_args.repeat):
                # noinspection PyTypeChecker
                evaluation = PartitionedEvaluation(learner=self._learner,
                                                   train_partitions=parsed_args.train_partitions,
                                                   eval_partitions=parsed_args.eval_partitions,
                                                   upsample=parsed_args.upsample,
                                                   majority_vote=parsed_args.majority_vote)
                evaluation.run(data_set)

                accuracies.append(evaluation.accuracy)
                uars.append(evaluation.uar)
                confusion_matrices.append(evaluation.confusion_matrix)

            accuracy = np.mean(accuracies)
            uar = np.mean(uars)

            # noinspection PyTypeChecker,PyStringFormat
            self.log.info("accuracy on %s: %2.2f%% (UAR %2.2f%%)" % (
                " & ".join([p.name for p in parsed_args.eval_partitions]), 100 * accuracy, 100 * uar))

        confusion_matrix = np.sum(confusion_matrices, axis=0)

        formatter = ConfusionMatrixFormatter()
        self.log.info("confusion matrix:\n%s", formatter.format(confusion_matrix, data_set.label_map))

        if self.app_args.verbose_level == 0:
            # support for piping the output of this command
            # noinspection PyStringFormat
            print("%.4f,%.4f" % (accuracy, uar))
        else:
            self.plot_confusion_matrix(confusion_matrix, sorted(data_set.label_map.keys()), normalize=True)


class MLPEvaluation(EvaluateBaseCommand):
    """
    Evaluate classification accuracy on generated features using an MLP
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
        checkpoint_dir = parsed_args.input.parent / "evaluation"  # type: Path

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        return TensorflowMLPLearner(checkpoint_dir=checkpoint_dir,
                                    num_layers=parsed_args.num_layers,
                                    num_units=parsed_args.num_units,
                                    learning_rate=parsed_args.learning_rate,
                                    num_epochs=parsed_args.num_epochs,
                                    keep_prob=parsed_args.keep_prob,
                                    shuffle_training=parsed_args.shuffle)


class SVMEvaluation(EvaluateBaseCommand):
    """
    Evaluate classification accuracy on generated features using an SVM
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
