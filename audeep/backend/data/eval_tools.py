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

"""Tools for evaluation, such as synthesizing a partitioning or cross-validation setup"""
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from audeep.backend.data.data_set import DataSet, Split, Partition


def create_cv_setup(data_set: DataSet,
                    num_folds: int) -> DataSet:
    """
    Add a randomly created cross-validation setup to the specified data set.
    
    f the specified data set contains multiple chunks per filename, chunks from the same filename are always placed in
    the same cross-validation split. If there additionally is full label information available, this method ensures 
    that classes are balanced between folds. Please note, however, that this method does not take into account any 
    further requirements such as ensuring that samples from the same original recording are placed in the same split, if 
    that original recording has been split into multiple audio files.
    
    Parameters
    ----------
    data_set: DataSet
        The data set to which a cross-validation setup should be added
    num_folds: int
        The number of cross-validation folds to create

    Returns
    -------
    DataSet
        A copy of the specified data set, with a cross-validation setup
    """
    log = logging.getLogger(__name__)

    data_set = data_set.with_cv_folds(num_folds).shuffled()

    if num_folds == 0:
        data_set.freeze()

        return data_set

    if data_set.is_fully_labeled:
        log.info("label information available - balancing classes between folds")

        # use pandas to get indices of instances of the same filename
        df = pd.DataFrame({"filenames": data_set.filenames})

        chunk_indices = [indices.tolist() for indices in df.groupby(df.filenames).groups.values()]

        # in a valid data set, all chunks of the same filename have the same label, and there is at least one chunk
        # per filename
        chunk_labels = np.array([data_set.labels_numeric[indices[0]] for indices in chunk_indices])

        labels, count = np.unique(chunk_labels, return_counts=True)
        label_indices = {l: [np.nonzero(chunk_labels == l)[0][i::num_folds] for i in range(num_folds)] for l in
                         labels}

        for l in label_indices:
            for fold, fold_indices in enumerate(label_indices[l]):
                cv_folds = [Split.TRAIN] * num_folds
                cv_folds[fold] = Split.VALID

                for chunk_index in fold_indices.tolist():
                    for index in chunk_indices[chunk_index]:
                        data_set[index].cv_folds = cv_folds
    else:
        log.info("no label information available - randomly splitting into folds")

        # use pandas to get indices of instances of the same filename
        df = pd.DataFrame({"filenames": data_set.filenames})

        chunk_indices = [indices.tolist() for indices in df.groupby(df.filenames).groups.values()]

        valid_split_indices = [chunk_indices[i::num_folds] for i in range(num_folds)]

        for fold, fold_indices in enumerate(valid_split_indices):
            cv_folds = [Split.TRAIN] * num_folds
            cv_folds[fold] = Split.VALID

            for chunk_index in fold_indices:
                for index in chunk_indices[chunk_index]:
                    data_set[index].cv_folds = cv_folds

    data_set.freeze()

    return data_set


def create_partitioning(data_set: DataSet,
                        partitions: Sequence[Partition]):
    """
    Add a randomly created partitioning setup to the specified data set.
    
    If the specified data set contains multiple chunks per filename, chunks from the same filename are always placed in
    the same partition. If there additionally is full label information available, this method ensures that classes are 
    balanced between partitions. Please note, however, that this method does not take into account any further 
    requirements such as ensuring that samples from the same original recording are placed in the same partition, if 
    that original recording has been split into multiple audio files.
   
    Parameters
    ----------
    data_set: DataSet
        The data set to which a partitioning setup should be added
    partitions: list of Partition
        The partitions which should be created.
   
    Returns
    -------
    DataSet
        A copy of the specified data set, with a partitioning setup
    """
    log = logging.getLogger(__name__)

    data_set = data_set.copy().shuffled()

    num_partitions = len(partitions)

    if data_set.is_fully_labeled:
        log.info("label information available - balancing classes between partitions")

        # use pandas to get indices of instances of the same filename
        df = pd.DataFrame({"filenames": data_set.filenames})

        chunk_indices = [indices.tolist() for indices in df.groupby(df.filenames).groups.values()]

        # in a valid data set, all chunks of the same filename have the same label, and there is at least one chunk
        # per filename
        chunk_labels = np.array([data_set.labels_numeric[indices[0]] for indices in chunk_indices])

        labels, count = np.unique(chunk_labels, return_counts=True)
        label_indices = {l: [np.nonzero(chunk_labels == l)[0][i::num_partitions] for i in range(num_partitions)] for l
                         in
                         labels}

        for l in label_indices:
            for partition_index, indices in enumerate(label_indices[l]):
                for index in indices:
                    data_set[index].partition = partitions[partition_index]
    else:
        log.info("no label information available - randomly splitting into partitions")

        # use pandas to get indices of instances of the same filename
        df = pd.DataFrame({"filenames": data_set.filenames})

        chunk_indices = [indices.tolist() for indices in df.groupby(df.filenames).groups.values()]

        partition_indices = [chunk_indices[i::num_partitions] for i in range(num_partitions)]

        for partition_index, indices in enumerate(partition_indices):
            for index in indices:
                data_set[index].partition = partitions[partition_index]

    data_set.freeze()

    return data_set
