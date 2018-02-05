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

"""Upsample partitioned data"""
import logging
from typing import Union, Sequence, Mapping

import numpy as np

from audeep.backend.data.data_set import DataSet, Partition, empty


def _invert_label_map(label_map: Mapping[str, int]) -> Mapping[int, str]:
    """
    Invert the label map of a data set.
    
    Since we know that label maps are always bijective, no additional checks have to be performed.
    
    Parameters
    ----------
    label_map: map of str to int
        The label map of a data set
        
    Returns
    -------
    map of int to str
        The inverted label map, which maps numeric label values to nominal label values
    """
    # noinspection PyTypeChecker
    return dict(map(reversed, label_map.items()))


def upsample(data_set: DataSet,
             partitions: Union[Partition, Sequence[Partition]] = None) -> DataSet:
    """
    Balance classes in the specified partitions of the specified data set.
    
    If `partitions` is set, instances in the specified partitions are repeated so that each class has approximately the 
    same number of instances. Any partitions present in the data set, but not specified as parameters to this function 
    are left unchanged.
    
    If `partitions` is empty or None, the entire data set is upsampled.
    
    If an instance is upsampled, the string "upsampled.I", where I indicates the repetition index, is appended to the
    filename.
    
    Parameters
    ----------
    data_set: DataSet
        The data set in which classes should be balanced
    partitions: Partition or list of Partition
        The partitions in which classes should be balanced

    Returns
    -------
    DataSet
        A new data set in which the classes in the specified partitions are balanced
    """
    log = logging.getLogger(__name__)

    if isinstance(partitions, Partition):
        partitions = [partitions]

    inverse_label_map = _invert_label_map(data_set.label_map)

    if partitions is None:
        keep_data = None
        upsample_data = data_set

        log.debug("upsampling entire data set")
    else:
        partitions_to_keep = [x for x in Partition if x not in partitions]

        # noinspection PyTypeChecker
        log.debug("upsampling partition(s) %s, keeping partition(s) %s", [x.name for x in partitions],
                  [x.name for x in partitions_to_keep])

        keep_data = None if not partitions_to_keep else data_set.partitions(partitions_to_keep)

        if keep_data is not None:
            upsample_data = data_set.partitions(partitions)
        else:
            upsample_data = data_set

    labels = upsample_data.labels_numeric
    unique, unique_count = np.unique(labels, return_counts=True)

    upsample_factors = np.max(unique_count) // unique_count

    num_instances = (0 if keep_data is None else keep_data.num_instances) + np.sum(upsample_factors * unique_count)

    log.info("upsampling with factors %s for labels %s, resulting in %d instances total", upsample_factors,
             [inverse_label_map[x] for x in unique], num_instances)

    upsample_map = dict(zip(unique, upsample_factors))

    # noinspection PyTypeChecker
    new_data = empty(num_instances, list(zip(data_set.feature_dims, data_set.feature_shape)), data_set.num_folds)
    new_data.label_map = data_set.label_map

    new_index = 0

    if keep_data is not None:
        # just copy instances we are not upsampling
        for index in keep_data:
            new_instance = new_data[new_index]
            old_instance = keep_data[index]

            new_instance.filename = old_instance.filename
            new_instance.chunk_nr = old_instance.chunk_nr
            new_instance.label_nominal = old_instance.label_nominal
            new_instance.cv_folds = old_instance.cv_folds
            new_instance.partition = old_instance.partition
            new_instance.features = old_instance.features

            new_index += 1

    for index in upsample_data:
        old_instance = upsample_data[index]

        for i in range(upsample_map[old_instance.label_numeric]):
            # repeat instance according to upsampling factor for the respective class
            new_instance = new_data[new_index]

            new_instance.filename = old_instance.filename + ".upsampled.%d" % (i + 1)
            new_instance.chunk_nr = old_instance.chunk_nr
            new_instance.label_nominal = old_instance.label_nominal
            new_instance.cv_folds = old_instance.cv_folds
            new_instance.partition = old_instance.partition
            new_instance.features = old_instance.features

            new_index += 1

    return new_data
