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

"""Main data handling implementation using xarray"""
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Tuple, Sequence, Type, Mapping, Optional, Union, Iterable, List

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from audeep.backend.log import LoggingMixin

_MISSING_VALUE_STR = "__MISSING_VALUE"
_MISSING_VALUE_INT = -999999


class _Dimension:
    """
    Dimension names used in the backing xarray dataset. 
    
    For internal use only.
    """
    INSTANCE = "instance"
    FOLD = "fold"


class _DataVar:
    """
    Names of the data variables in the backing xarray dataset. 
    
    For internal use only.
    """
    FILENAME = "filename"
    CHUNK_NR = "chunk_nr"
    LABEL_NOMINAL = "label_nominal"
    LABEL_NUMERIC = "label_numeric"
    CV_FOLDS = "cv_folds"
    PARTITION = "partition"
    FEATURES = "features"


class _Attribute:
    """
    Names of metadata attributes in the backing xarray dataset. 
    
    For internal use only.
    """
    FEATURE_DIMS = "feature_dims"
    LABEL_MAP = "label_map"


class Split(Enum):
    """
    Identifiers for cross validation splits.
    """
    TRAIN = 0
    VALID = 1


class Partition(Enum):
    """
    Identifiers for different data partitions.
    """
    TRAIN = 0
    DEVEL = 1
    TEST = 2


class _Instance:
    """
    Represents a single instance in a data set.
    
    This class is returned by the indexing methods on the DataSet class, but should never be instantiated outside of 
    this module. Instance data is not copied from the backing xarray data set. Instead, this class implements a view
    on the xarray data set restricted to a single instance.
    """

    def __init__(self,
                 instance: int,
                 data: xr.Dataset,
                 mutable: bool = False):
        """
        Create a new _Instance view representing the specified instance of the specified xarray data set.
        
        Parameters
        ----------
        instance: int
            The index of the instance in the specified xarray data set
        data: xarray.Dataset
            The xarray data set containing the instance
        mutable: bool, optional
            If True, attributes of this instance may be modified. If False (default), any attempt to modify the instance
            will result in an AttributeError
        """
        self._instance = instance
        self._data = data
        self._mutable = mutable

    def _get_value(self, var: str):
        """
        Utility method to return the value of the specified variable for this instance in the backing xarray data set.
        
        Parameters
        ----------
        var: str
            Name of the variable. There should be no reason to pass a str directly. Instead, the names defined in the
            _DataVar class should be used.

        Returns
        -------
        depending on variable
            The value of the specified variable for this instance
        """
        return self._data[var][dict(instance=self._instance)]

    def _set_value(self, var: str, value):
        """
        Utility method to set the value of the specified variable for this instance in the backing xarray data set.
        
        This method does **not** check whether the instance is immutable.
        
        Parameters
        ----------
        var: str
            Name of the variable. There should be no reason to pass a str directly. Instead, the names defined in the
            _DataVar class should be used.
        value: depending on variable
            New value for the variable
        """
        self._data[var][dict(instance=self._instance)] = value

    @property
    def feature_dims(self) -> Sequence[str]:
        """
        Returns the names of the feature dimensions, in the order in which they are stored.
        
        Returns
        -------
        list of str
            The names of the feature dimensions, in the order in which they are stored
        """
        return json.loads(self._data.attrs[_Attribute.FEATURE_DIMS])

    @property
    def label_map(self) -> Mapping[str, int]:
        """
        Returns the label map, if defined.
        
        Returns
        -------
        map of str to int
            The label map, if defined
        """
        return dict(json.loads(self._data.attrs[_Attribute.LABEL_MAP])) \
            if _Attribute.LABEL_MAP in self._data.attrs else None

    @property
    def feature_shape(self) -> Sequence[int]:
        """
        Returns the shape of the feature matrix of this instance.
        
        The entries in the returned shape correspond to the dimensions returned by the `feature_dims` property.
        
        Returns
        -------
        list of int
            The size of the feature matrix dimensions
        """
        return [self._data.dims[x] for x in self.feature_dims]

    @property
    def filename(self) -> str:
        """
        Returns the filename of this instance.
        
        In a valid data set, the filename is always defined.
        
        Returns
        -------
        str
            The filename of the instance
        """
        result = self._get_value(_DataVar.FILENAME).values.tolist()

        return None if result is None else str(result)

    @filename.setter
    def filename(self, value: str):
        """
        Sets the filename of this instance to the specified value.
        
        Parameters
        ----------
        value: str
            The new filename
        
        Raises
        ------
        AttributeError
            If the instance is immutable
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")

        self._set_value(_DataVar.FILENAME, value)

    @property
    def chunk_nr(self) -> int:
        """
        Returns the chunk number of this instance.
        
        In a valid data set, the chunk number is always defined.
        
        Returns
        -------
        int
            The chunk number of this instance
        """
        result = self._get_value(_DataVar.CHUNK_NR).values.tolist()

        return None if result is None else int(result)

    @chunk_nr.setter
    def chunk_nr(self, value: int):
        """
        Sets the chunk number of this instance to the specified value.
        
        Parameters
        ----------
        value: int
            The new chunk number

        Raises
        ------
        AttributeError
            If the instance is immutable
        ValueError
            If the chunk number is less than zero
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")
        if not 0 <= value:
            raise ValueError("invalid chunk number: {}".format(value))

        self._set_value(_DataVar.CHUNK_NR, value)

    @property
    def label_nominal(self) -> str:
        """
        Returns the nominal label of this instance.
        
        Returns
        -------
        str
            The nominal label of this instance
        """
        result = self._get_value(_DataVar.LABEL_NOMINAL).values.tolist()

        return None if result is None else str(result)

    @label_nominal.setter
    def label_nominal(self, value: str):
        """
        Sets the nominal label of this instance to the specified value.
        
        If a label map is defined for the data set, the specified value must be a valid nominal label according to the
        label map. Furthermore, the numeric label is automatically set to the numeric value of the nominal label as
        specified by the label map.
        
        Parameters
        ----------
        value: str
            The new nominal label
            
        Raises
        ------
        AttributeError
            If the instance is immutable
        ValueError
            If a label map is given and the specified nominal label is not a key in the label map

        """
        if not self._mutable:
            raise AttributeError("data set is immutable")
        if self.label_map is not None and value is not None and value not in self.label_map:
            raise ValueError(
                "nominal label not contained in label map: {} not in {}".format(value, self.label_map.keys()))

        self._set_value(_DataVar.LABEL_NOMINAL, value)

        if self.label_map is not None:
            if value is None:
                self._set_value(_DataVar.LABEL_NUMERIC, None)
            else:
                # noinspection PyTypeChecker
                self._set_value(_DataVar.LABEL_NUMERIC, self.label_map[value])

    @property
    def label_numeric(self) -> int:
        """
        Returns the numeric label of this instance.
        
        Returns
        -------
        int
            The numeric label of this instance
        """
        return self._get_value(_DataVar.LABEL_NUMERIC).values.tolist()

    @label_numeric.setter
    def label_numeric(self, value: int):
        """
        Sets the numeric label of this instance to the specified value.
        
        If a label map is defined for the data set, the numeric label cannot be set directly. Instead, the nominal label
        should be set, which will automatically update the numeric label.
        
        Parameters
        ----------
        value: int
            The new numeric label

        Raises
        ------
        AttributeError
            If the instance is immutable or a label map is given
        ValueError
            If the numeric label happens to clash with the reserved value used to represent missing values. Since
            numeric labels should generally be positive numbers, and the reserved value is negative, this should not
            happen in practice.
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")
        if self.label_map is not None:
            raise AttributeError("cannot set numeric labels directly, since label map is given")
        if value == _MISSING_VALUE_INT:
            raise ValueError("cannot use reserved value {}".format(_MISSING_VALUE_INT))

        self._set_value(_DataVar.LABEL_NUMERIC, None if value is None else value)

    @property
    def cv_folds(self) -> Sequence[Split]:
        """
        Returns the cross-validation information associated with this instance.
        
        For each cross-validation fold, if any, the split to which this instance belongs is returned.
        
        Returns
        -------
        list of Split
            The split to which this instance belongs for each cross-validation fold
        """
        splits = self._get_value(_DataVar.CV_FOLDS).values

        return [None if x is None else Split(x) for x in splits]

    @cv_folds.setter
    def cv_folds(self, value: Sequence[Split]):
        """
        Sets the cross-validation splits for this instance to the specified value.
        
        Parameters
        ----------
        value: list of Split
            The new cross-validation splits for each fold. Must contain one entry for each cross-validation fold
            
        Raises
        ------
        AttributeError
            If the instance is immutable
        ValueError
            If the specified list of cross-validation splits does not have exactly one entry for each fold
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")

        if value is None:
            value = []

        if len(value) != self._data.dims[_Dimension.FOLD]:
            raise ValueError("invalid number of folds: expected {}, got {}"
                             .format(self._data.dims[_Dimension.FOLD], len(value)))

        numeric_value = [None if x is None else x.value for x in value]

        self._set_value(_DataVar.CV_FOLDS, numeric_value)

    @property
    def partition(self) -> Partition:
        """
        Returns the partition to which this instance belongs.
        
        Returns
        -------
        Partition
            The partition to which this instance belongs
        """
        partition = self._get_value(_DataVar.PARTITION).values.tolist()

        return None if partition is None else Partition(partition)

    @partition.setter
    def partition(self, value: Partition):
        """
        Sets the partition to which this instance belongs to the specified value
        
        Parameters
        ----------
        value: Partition
            The new partition
            
        Raises
        ------
        AttributeError
            If the instance is immutable
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")

        numeric_value = None if value is None else value.value

        self._set_value(_DataVar.PARTITION, numeric_value)

    @property
    def features(self) -> np.ndarray:
        """
        Returns the feature matrix of this instance.
        
        Returns
        -------
        numpy.ndarray
            The feature matrix of this instance
        """
        return self._get_value(_DataVar.FEATURES).values

    @features.setter
    def features(self, value: np.ndarray):
        """
        Sets the feature matrix of this instance to the specified value.
        
        The specified feature matrix must have the correct shape, as specified by the `feature_shape` property, and must
        not contain NaN values.
        
        Parameters
        ----------
        value: numpy.ndarray
            The new feature matrix
            
        Raises
        ------
        AttributeError
            If the instance is immutable
        ValueError
            If the specified feature matrix has invalid shape or contains NaN values
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")
        if list(value.shape) != self.feature_shape:
            raise ValueError("invalid feature shape: expected {}, got {}".format(self.feature_shape, list(value.shape)))
        if np.any(np.isnan(value)):
            raise ValueError("features may not contain nan values")

        self._set_value(_DataVar.FEATURES, value.astype(np.float32))

    def __str__(self) -> str:
        """
        Returns a string representation of this instance.
        
        Returns
        -------
        str
            A string representation of this instance
        """
        return self._data[dict(instance=self._instance)].__str__()


class DataSet(LoggingMixin):
    """
    A data set containing instances which each have certain metadata and numeric features.
    
    Instances can have feature matrices of arbitrary dimensionality and shape, but dimensionality and shape must be the
    same for all instances. Most metadata is not required to be present, so that data sets with missing or partial
    metadata can still be represented. Depending on which metadata is present, several high-level operations such as
    splitting the data set by partitions are supported.
    """

    def __init__(self,
                 data: xr.Dataset,
                 mutable: bool = False):
        """
        Create and initialize a new DataSet with the specified parameters.
        
        There should be no reason to invoke this constructor directly. Instead, the utility methods for loading a data
        set from a file, or for creating an empty data set should be used.
        
        Parameters
        ----------
        data: xarray.Dataset
            The xarray data set storing the actual data
        mutable: bool
            True, if modifications to the data set should be allowed, False otherwise
        """
        super().__init__()

        self._data = data
        self._mutable = mutable

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances in this data set.
        
        Returns
        -------
        int
            The number of instances in this data set
        """
        return self._data.dims[_Dimension.INSTANCE]

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds in this data set.
        
        Returns
        -------
        int
            The number of cross-validation folds in this data set
        """
        return self._data.dims[_Dimension.FOLD]

    @property
    def feature_dims(self) -> Sequence[str]:
        """
        Returns the names of the feature dimensions, in the order in which they are stored.
        
        Returns
        -------
        list of str
            The names of the feature dimensions, in the order in which they are stored
        """
        return json.loads(self._data.attrs[_Attribute.FEATURE_DIMS])

    @property
    def feature_shape(self) -> Sequence[int]:
        """
        Returns the shape of the feature matrix of this instance.

        The entries in the returned shape correspond to the dimensions returned by the `feature_dims` property.

        Returns
        -------
        list of int
            The size of the feature matrix dimensions
        """
        return [self._data.dims[x] for x in self.feature_dims]

    @property
    def has_cv_info(self) -> bool:
        """
        Checks whether any cross-validation information is associated with this data set. 
        
        In order for this property to be True, there must be at least one fold, and no missing split values for any 
        fold and any instance.
        
        Returns
        -------
        bool
            True, if all instances are assigned a cross-validation split for each fold, False otherwise
        """
        return self.num_folds > 0 and not self._data[_DataVar.CV_FOLDS].isnull().any()

    @property
    def has_partition_info(self) -> bool:
        """
        Checks whether any partition information is associated with this data set. 
        
        In order for this property to be true, there must be a valid partition assigned to each instance.
        
        Returns
        -------
        bool
            True, if all instances are assigned a partition, False otherwise
        """
        return not self._data[_DataVar.PARTITION].isnull().any()

    @property
    def is_fully_labeled(self) -> bool:
        """
        Checks whether this data set is fully labeled. 
        
        A data set is fully labeled if both nominal and numeric labels are specified for each instance.
        
        Returns
        -------
        bool
            True, if this data set is fully labeled, False otherwise
        """
        return not self._data[_DataVar.LABEL_NOMINAL].isnull().any() and \
               not self._data[_DataVar.LABEL_NUMERIC].isnull().any()

    @property
    def has_overlapping_folds(self) -> bool:
        """
        Checks whether this data set has overlapping folds.
        
        A data set has overlapping folds if there are instances which belong to the validation split in multiple folds.
        If the data set does not have full cross-validation information, this property returns False.
        
        Returns
        -------
        bool
            True, if this data set has full cross-validation information and overlapping folds, False otherwise
        """
        if not self.has_cv_info:
            return False

        for index in self:
            if self[index].cv_folds.count(Split.VALID) != 1:
                return True

        return False

    def contains(self,
                 filename: str,
                 chunk_nr: int) -> bool:
        """
        Check whether this data set contains an instance with the specified filename and chunk number.
        
        Parameters
        ----------
        filename: str
            The filename of the instance
        chunk_nr: int
            The chunk number of the instance

        Returns
        -------
        bool
            True, if this data set contains an instance with the specified filename and chunk number, False otherwise
        """
        if filename not in self._data[_DataVar.FILENAME].values:
            return False

        instances_with_filename = self._data.where(self._data[_DataVar.FILENAME] == filename)

        return chunk_nr in instances_with_filename[_DataVar.CHUNK_NR].values

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns the label map, if defined.

        The label map defines a mapping of nominal label values to numeric label values. If a label map is given,
        consistency of the labels is enforced.

        Returns
        -------
        map of str to int
            The label map, if defined
        """
        if _Attribute.LABEL_MAP not in self._data.attrs:
            return None

        return dict(json.loads(self._data.attrs[_Attribute.LABEL_MAP]))

    @label_map.setter
    def label_map(self, value: Mapping[str, int]):
        """
        Sets the label map to the specified value.
        
        Please not that **no** validation is performed whether the current labels are consistent with the specified
        label map.
        
        Parameters
        ----------
        value: map of str to int
            The new label map
            
        Raises
        ------
        AttributeError
            If the data set is immutable
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")

        if value is None:
            if _Attribute.LABEL_MAP in self._data.attrs:
                del self._data.attrs[_Attribute.LABEL_MAP]
        else:
            # sort dictionary items to make sure that equal label maps have equal representations
            self._data.attrs[_Attribute.LABEL_MAP] = json.dumps(sorted(value.items(), key=lambda i: i[0]))

    @property
    def filenames(self) -> np.ndarray:
        """
        Returns the filenames of all instances in this data set as a NumPy array.
        
        The order of filenames in the returned array matches the order in which instances are stored in this data set.
        
        Returns
        -------
        numpy.ndarray
            The filenames of all instances in this data set as a NumPy array
        """

        return self._data[_DataVar.FILENAME].values.astype(np.str)

    @property
    def labels_numeric(self) -> np.ndarray:
        """
        Returns the numeric labels of all instances in this data set as a NumPy array.
        
        The order of labels in the returned array matches the order in which instances are stored in this data set.
        
        Returns
        -------
        numpy.ndarray
            The numeric labels of the instances in this data set
            
        Raises
        ------
        AttributeError
            If the data set is not fully labeled
        """
        if not self.is_fully_labeled:
            raise AttributeError("data set does not have label information")

        return self._data[_DataVar.LABEL_NUMERIC].values.astype(np.int)

    @property
    def labels_nominal(self) -> np.ndarray:
        """
        Returns the nominal labels of all instances in this data set as a NumPy array.
        
        The order of labels in the returned array matches the order in which instances are stored in this data set.
        
        Returns
        -------
        numpy.ndarray
            The nominal labels of the instances in this data set
            
        Raises
        ------
        AttributeError
            If the data set is not fully labeled
        """
        if not self.is_fully_labeled:
            raise AttributeError("data set does not have label information")

        return self._data[_DataVar.LABEL_NOMINAL].values.astype(np.str)

    @property
    def filename_labels_numeric(self) -> Mapping[str, int]:
        """
        Returns the numeric labels of each audio file.
        
        If the data set contains only one chunk per audio file, this function returns semantically the same data as 
        the `labels_numeric` function. If, however, the data set contains multiple chunks per audio file, this function
        returns only one entry for each audio file. No information is lost, since in a valid data set, all chunks of the
        same original instance must have the same label.
        
        Returns
        -------
        map of str to int
            A mapping of filenames to numeric labels. 
        """
        if not self.is_fully_labeled:
            raise AttributeError("data set does not have label information")

        labels = {}

        for index in self:
            instance = self[index]
            labels[instance.filename] = instance.label_numeric

        return labels

    @property
    def features(self) -> np.ndarray:
        """
        Returns the feature matrices of all instances in this data set as a NumPy array.
        
        The first axis of the returned array corresponds to different instances. The remaining axes match the dimensions
        returned by the `feature_dims` property.
        
        Returns
        -------
        numpy.ndarray
            The feature matrices of all instances in this data set
        """
        return self._data[_DataVar.FEATURES].values

    @features.setter
    def features(self, value: np.ndarray):
        """
        Sets the feature matrices of all instances to the specified value.
        
        The first axis of the specified NumPy array must have one entry for each instance, the remaining axes must match
        the shape returned by the `feature_shape` property.
        
        Parameters
        ----------
        value: numpy.ndarray
            The new feature matrices for all instances

        Raises
        ------
        AttributeError
            If the data set is immutable
        ValueError
            If the specified feature matrices have invalid shape or contain NaN values
        """
        if not self._mutable:
            raise AttributeError("data set is immutable")
        # noinspection PyTypeChecker
        if list(value.shape) != [self.num_instances] + self.feature_shape:
            # noinspection PyTypeChecker
            raise ValueError("invalid feature shape: expected {}, got {}"
                             .format([self.num_instances] + self.feature_shape, list(value.shape)))
        if np.isnan(value).any():
            raise ValueError("features may not contain nan")

        # noinspection PyTypeChecker
        self._data[_DataVar.FEATURES] = ([_Dimension.INSTANCE] + self.feature_dims, value.astype(np.float32))

    def partitions(self,
                   partitions: Union[Partition, Sequence[Partition]]):
        """
        Returns a new data set containing only the instances in the specified partitions
        
        Parameters
        ----------
        partitions: Partition or list of Partition
            The partitions to include in the new data set
            
        Returns
        -------
        DataSet
            A new data set containing only the instances in the specified partitions
            
        Raises
        ------
        ValueError
            If there are instances with missing partition information
        """
        if not self.has_partition_info:
            raise ValueError("data set does not have partition info")

        if isinstance(partitions, Partition):
            partitions = [partitions]

        # noinspection PyTypeChecker
        conds = [self._data[_DataVar.PARTITION] == x.value for x in partitions]
        or_cond = xr.full_like(conds[0], fill_value=False, dtype=np.bool)

        for cond in conds:
            # noinspection PyUnresolvedReferences
            or_cond = xr.ufuncs.logical_or(or_cond, cond)

        new_data = self._data.where(or_cond, drop=True)

        return DataSet(data=new_data,
                       mutable=self._mutable)

    def split(self,
              fold: int,
              split: Split):
        """
        Returns a new data set containing only the instances which belong to the specified split in the specified
        cross-validation fold.
        
        Parameters
        ----------
        fold: int
            The fold index
        split: Split
            The split

        Returns
        -------
        DataSet
            A new data set containing only the instances which belong to the specified split in the specified 
            cross-validation fold
            
        Raises
        ------
        ValueError
            If there are instances with missing cross-validation information
        IndexError
            If the specified fold index is out of range
        """
        if not self.has_cv_info:
            raise ValueError("data set does not have cross validation info")
        if not 0 <= fold < self.num_folds:
            raise IndexError("invalid fold index: {}".format(fold))

        new_data = self._data.where(self._data[dict(fold=fold)][_DataVar.CV_FOLDS] == split.value, drop=True)

        return DataSet(data=new_data,
                       mutable=self._mutable)

    def freeze(self):
        """
        Makes this data set immutable.
        """
        self._mutable = False

    def copy(self):
        """
        Returns a mutable copy of this data set.

        Returns
        -------
        DataSet
            A mutable copy of this data set
        """
        return DataSet(data=self._data.copy(),
                       mutable=True)

    def with_feature_dimensions(self,
                                feature_dimensions: Sequence[Tuple[str, int]]):
        """
        Returns a new data set with the same instance metadata as this data set, and the specified feature dimensions.
        
        The returned data set is mutable, and has all missing values in the feature matrices. The specified feature
        dimensions must be a list of tuples, where each tuple contains the name followed by the size of one feature
        dimension.
        
        Parameters
        ----------
        feature_dimensions: list of tuple of str and int
            The feature dimensions of the new data set
            
        Returns
        -------
        DataSet
            A new data set with the same instance metadata as this data set, and the specified feature dimensions
        """
        result = empty(num_instances=self.num_instances,
                       feature_dimensions=feature_dimensions,
                       num_folds=self.num_folds)

        result._data[_DataVar.FILENAME] = self._data[_DataVar.FILENAME].copy()
        result._data[_DataVar.CHUNK_NR] = self._data[_DataVar.CHUNK_NR].copy()
        result._data[_DataVar.LABEL_NOMINAL] = self._data[_DataVar.LABEL_NOMINAL].copy()
        result._data[_DataVar.LABEL_NUMERIC] = self._data[_DataVar.LABEL_NUMERIC].copy()
        result._data[_DataVar.CV_FOLDS] = self._data[_DataVar.CV_FOLDS].copy()
        result._data[_DataVar.PARTITION] = self._data[_DataVar.PARTITION].copy()
        result.label_map = self.label_map

        return result

    def with_cv_folds(self,
                      num_folds: int):
        """
        Returns a new data set containing the same data as this data set and the specified number of cross validation 
        folds.
        
        The new data set will have no cross-validation information, even if the current data set does.
        
        Parameters
        ----------
        num_folds: int
            The desired number of cross-validation folds

        Returns
        -------
        DataSet
            A new data set containing the same data as this data set and the specified number of cross validation folds
        """
        result = empty(num_instances=self.num_instances,
                       feature_dimensions=list(zip(self.feature_dims, self.feature_shape)),
                       num_folds=num_folds)

        result._data[_DataVar.FILENAME] = self._data[_DataVar.FILENAME].copy()
        result._data[_DataVar.CHUNK_NR] = self._data[_DataVar.CHUNK_NR].copy()
        result._data[_DataVar.LABEL_NOMINAL] = self._data[_DataVar.LABEL_NOMINAL].copy()
        result._data[_DataVar.LABEL_NUMERIC] = self._data[_DataVar.LABEL_NUMERIC].copy()
        result._data[_DataVar.PARTITION] = self._data[_DataVar.PARTITION].copy()
        result._data[_DataVar.FEATURES] = self._data[_DataVar.FEATURES].copy()
        result.label_map = self.label_map

        return result

    def shuffled(self):
        """
        Returns a new data set containing the same instances as this data set in random order.
        
        Returns
        -------
        DataSet
            A new data set containing the same instances as this data set in random order
        """
        perm = np.random.permutation(self.num_instances)

        new_data = self._data.copy()
        new_data = new_data[{_Dimension.INSTANCE: perm}]

        result = DataSet(data=new_data,
                         mutable=True)
        result.label_map = self.label_map

        if not self._mutable:
            result.freeze()

        return result

    def scaled(self,
               scaler: StandardScaler):
        """
        Returns a new data set with features transformed by the specified scaler.
        
        Parameters
        ----------
        scaler: StandardScaler
            A sklearn standardization scaler
            
        Returns
        -------
        DataSet
            The transformed data set
        """
        new_data = self._data.copy()

        result = DataSet(data=new_data,
                         mutable=True)
        result.label_map = self.label_map
        result.features = scaler.transform(result.features)

        if not self._mutable:
            result.freeze()

        return result

    def save(self, path: Path):
        """
        Writes this data set to the specified path.
        
        Any directories in the path that do not exist are automatically created.
        
        Parameters
        ----------
        path: pathlib.Path
        """
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        self.log.info("writing data set as netCDF4 to %s", path)

        self._data.to_netcdf(path=str(path),
                             engine="netcdf4",
                             format="NETCDF4")

    def __getitem__(self, item):
        """
        Returns the instance at the specified index.
        
        Parameters
        ----------
        item: int
            The index of the instance that should be retrieved

        Returns
        -------
        _Instance
            The index at the specified index
            
        Raises
        ------
        IndexError
            If the specified index is smaller than zero or greater than or equal to the number of instance in this data 
            set
        """
        if not 0 <= item < self.num_instances:
            raise IndexError("invalid instance index: {}".format(item))

        return _Instance(instance=item,
                         data=self._data,
                         mutable=self._mutable)

    def __iter__(self):
        """
        Iterates over the instance indices in this data set.
        
        Returns
        -------
        generator of int
            A generator for the instance indices in this data set
        """
        return (x for x in range(self.num_instances))

    def __str__(self):
        """
        Returns a string representation of this data set.
        
        This method returns a string representation of the backing xarray Dataset.
        
        Returns
        -------
        str
            A string representation of this data set
        """
        return self._data.__str__()


def _empty_ndarray(shape: Sequence[int],
                   dtype: Type) -> np.ndarray:
    """
    Returns an empty numpy ndarray with the specified shape and data type.
    
    Depending on the data type, the array is filled with an appropriate placeholder for missing values. Currently, this
    method support numpy.object arrays, which will be filled with None values, and numpy.float* data types, which will
    be filled with numpy.NaN values.
    
    Parameters
    ----------
    shape: list of int
        The desired shape of the ndarray
    dtype: data type
        The desired data type of the ndarray

    Returns
    -------
    numpy.ndarray
        An empty ndarray of the specified shape and data type
    """
    result = np.empty(shape=shape, dtype=dtype)

    if dtype is np.object:
        result[:] = None
    elif dtype is np.float16 or dtype is np.float32 or dtype is np.float64:
        result[:] = np.NaN
    else:
        raise ValueError("unsupported data type: {}".format(dtype))

    return result


def empty(num_instances: int,
          feature_dimensions: Sequence[Tuple[str, int]],
          num_folds: int) -> DataSet:
    """
    Returns an empty data set created with the specified parameters.
    
    The data set will be filled with missing values, and will be mutable.
    
    Parameters
    ----------
    num_instances: int
        The number of instances in the data set
    feature_dimensions: list of tuple(str, int)
        A list of names and sizes for the feature dimensions. Each entry in the list must be a tuple containing the name
        of a feature dimension and the size of a feature dimension.
    num_folds: int
        The number of cross-validation folds in the data set. May be zero.

    Returns
    -------
    DataSet
        An empty data set created with the specified parameters
    """
    feature_dims = [dim[0] for dim in feature_dimensions]
    feature_shape = [num_instances] + [dim[1] for dim in feature_dimensions]

    # @formatter:off
    data = xr.Dataset({
        _DataVar.FILENAME: ([_Dimension.INSTANCE], _empty_ndarray([num_instances], dtype=np.object)),
        _DataVar.CHUNK_NR: ([_Dimension.INSTANCE], _empty_ndarray([num_instances], dtype=np.object)),
        _DataVar.CV_FOLDS: ([_Dimension.INSTANCE, _Dimension.FOLD],
                            _empty_ndarray([num_instances, num_folds], dtype=np.object)),
        _DataVar.PARTITION: ([_Dimension.INSTANCE], _empty_ndarray([num_instances], dtype=np.object)),
        _DataVar.LABEL_NOMINAL: ([_Dimension.INSTANCE], _empty_ndarray([num_instances], dtype=np.object)),
        _DataVar.LABEL_NUMERIC: ([_Dimension.INSTANCE], _empty_ndarray([num_instances], dtype=np.object)),
        _DataVar.FEATURES: ([_Dimension.INSTANCE] + feature_dims, _empty_ndarray(feature_shape, dtype=np.float32))
    })
    # @formatter:on

    data.attrs[_Attribute.FEATURE_DIMS] = json.dumps(feature_dims)

    return DataSet(data=data,
                   mutable=True)


def load(path: Path) -> DataSet:
    """
    Loads a data set from the specified NetCDF4 file.
    
    Parameters
    ----------
    path: pathlib.Path
        Path to the file which should be loaded.

    Returns
    -------
    DataSet
        The data set loaded from the specified file
    """
    log = logging.getLogger(__name__)
    log.info("loading data set from %s", path)

    data = xr.open_dataset(str(path))  # type: xr.Dataset

    # restore data types
    data[_DataVar.FILENAME] = data[_DataVar.FILENAME].astype(np.object).fillna(None)
    data[_DataVar.CHUNK_NR] = data[_DataVar.CHUNK_NR].astype(np.object).fillna(None)
    data[_DataVar.CV_FOLDS] = data[_DataVar.CV_FOLDS].astype(np.object).fillna(None)
    data[_DataVar.PARTITION] = data[_DataVar.PARTITION].astype(np.object).fillna(None)
    data[_DataVar.LABEL_NOMINAL] = data[_DataVar.LABEL_NOMINAL].astype(np.object).fillna(None)
    data[_DataVar.LABEL_NUMERIC] = data[_DataVar.LABEL_NUMERIC].astype(np.object)
    data[_DataVar.FEATURES] = data[_DataVar.FEATURES].astype(np.float32)

    return DataSet(data=data,
                   mutable=False)


def concat_instances(data_sets: Iterable[DataSet]) -> DataSet:
    """
    Concatenates the specified data sets along the instance dimension.
    
    All data sets must have exactly matching metadata.
    
    Parameters
    ----------
    data_sets: list of DataSet
        The data sets to concatenate

    Returns
    -------
    DataSet
        A new data set containing all instances in the specified data sets
    """
    # noinspection PyProtectedMember
    mutable = all([data_set._mutable for data_set in data_sets])

    filenames = np.concatenate([data_set._data[_DataVar.FILENAME].values for data_set in data_sets])

    if len(np.unique(filenames)) != len(filenames):
        raise ValueError("data sets contain duplicate instances - refusing to concatenate")

    # noinspection PyProtectedMember
    new_data = xr.concat([data_set._data for data_set in data_sets],
                         dim=_Dimension.INSTANCE,
                         data_vars="minimal",
                         compat="identical")

    # noinspection PyTypeChecker
    return DataSet(data=new_data,
                   mutable=mutable)


def concat_features(data_sets: Iterable[DataSet],
                    dimension: str = "generated") -> DataSet:
    """
    Concatenates the specified data sets along the specified feature dimension.
    
    The feature matrices of each instance are concatenated along the specified dimension. All data sets must have the 
    specified feature dimension. Additionally, all metadata, including names and sizes of other feature dimensions must 
    be identical for all data sets.
    
    Parameters
    ----------
    data_sets: list of DataSet
        Data sets which should be concatenated
    dimension: str, default "generated"
        Feature dimension along which features should be concatenated.
        
    Returns
    -------
    DataSet
        A new data set created by concatenating the specified data sets along the specified feature dimension
        
    Raises
    ------
    ValueError
        If the specified feature dimension is not present in some data sets
    """
    for data_set in data_sets:
        # noinspection PyProtectedMember
        if dimension not in data_set._data.dims:
            raise ValueError("dimension '{}' missing in some data sets".format(dimension))

    # noinspection PyProtectedMember
    mutable = all([data_set._mutable for data_set in data_sets])

    # noinspection PyProtectedMember
    new_data = xr.concat([data_set._data for data_set in data_sets],
                         dim=dimension,
                         data_vars="minimal",
                         compat="identical")

    # noinspection PyTypeChecker
    return DataSet(data=new_data,
                   mutable=mutable)


def concat_chunks(data_set: DataSet,
                  dimension: str = "generated") -> DataSet:
    """
    Concatenates chunked instances within the specified data set along the specified feature dimension.
    
    For each audio file, all chunks are collected in order and their feature matrices are concatenated along the
    specified dimension. For all chunks of the same instance, all metadata must be identical.
    
    Parameters
    ----------
    data_set: DataSet
        The data set in which chunks should be fused
    dimension: str
        The feature dimension along which instances should be concatenated

    Returns
    -------
    DataSet
        A new data set in which chunked instances have been fused
        
    Raises
    ------
    ValueError
        If the specified feature dimension does not exist in the specified data set
    """
    # noinspection PyProtectedMember
    if dimension not in data_set._data.dims:
        raise ValueError("invalid dimension: {}".format(dimension))

    # noinspection PyProtectedMember
    data = data_set._data.copy()

    # step 1: group instances by filename
    group_by_filename = data.groupby(_DataVar.FILENAME)

    # check whether all groups have the same number of instances
    num_chunks = None

    for _, filename_group in group_by_filename:  # type: xr.Dataset
        if num_chunks is None:
            num_chunks = filename_group.dims[_Dimension.INSTANCE]
        elif num_chunks != filename_group.dims[_Dimension.INSTANCE]:
            raise ValueError("data set contains an inconsistent number of chunks per file")

    filenames = []

    # step 2: for each group, group instances by chunk number, which should result in num_chunks singleton data sets
    for filename, filename_group in group_by_filename:  # type: xr.Dataset
        group_by_chunks = filename_group.groupby(_DataVar.CHUNK_NR)

        chunks = []

        for index, (chunk_nr, chunk_group) in enumerate(sorted(group_by_chunks, key=lambda x: x[0])):
            if chunk_group.dims[_Dimension.INSTANCE] != 1:
                raise ValueError("duplicate chunk number {} for file {}".format(index, filename))
            if chunk_nr != index:
                raise ValueError("inconsistent chunk numbering for file {}: expected {}, got {}"
                                 .format(filename, index, chunk_nr))

            # set chunk number to zero, since we want to concatenate later
            chunk_group[_DataVar.CHUNK_NR][{_Dimension.INSTANCE: 0}] = 0

            chunks.append(chunk_group)

        filenames.append(xr.concat(chunks, dim=dimension, data_vars="minimal", compat="identical"))

    new_data = xr.concat(filenames, dim=_Dimension.INSTANCE, data_vars="all")

    # noinspection PyTypeChecker,PyProtectedMember
    return DataSet(data=new_data,
                   mutable=data_set._mutable)


def _format_values(values: List,
                   missing_value):
    """
    Formats a list of values, which may contain missing values indicated by the specified placeholder.
    
    Returns a list containing a string representation for each value in the specified list, which is "None" if the 
    element value is equal to the specified missing value placeholder, or the actual value otherwise.
    
    Parameters
    ----------
    values: list
        A list of values, with missing value indicated by the specified placeholder value
    missing_value: Any
        A placeholder value for missing values

    Returns
    -------
    list
        A list of strings containing a string representation for each element in the specified list
    """
    return ["None" if x == missing_value else x for x in values]


def _format_enum_values(values: List[int],
                        enum_class: Type[Enum]):
    """
    Formats a list of enum values represented by their numeric value, which may contain missing values indicated by 
    None.
    
    Returns a list containing a string representation for each list element, which is "None" if the element is None, or
    the name of the enum member corresponding to the numeric value in the list otherwise.
    
    Parameters
    ----------
    values: list
        A list of numeric enum values
    enum_class: Type[Enum]
        The enum type

    Returns
    -------
    list
        A list of strings containing a string representation for each element in the specified list
    """
    return ["None" if x == _MISSING_VALUE_INT else enum_class(x).name for x in values]


def check_integrity(data_set: DataSet) -> Optional[Mapping[int, List[str]]]:
    """
    Check data set integrity according to the following constraints.
    
    1. Instances with the same filename must have the same nominal labels
    2. Instances with the same filename must have the same numeric labels
    3. Instances with the same filename must have the same cross validation information
    4. Instances with the same filename must have the same partition information
    5. If a label map is given, all nominal labels must be keys in the map, and all numeric labels must be the 
       associated values
    6. For each filename, there must be the same number of chunks
    7. For each filename, chunk numbers must be exactly [0, ..., num_chunks], i.e. each chunk number must be present
       exactly once.
       
    This is a potentially costly operation, and therefore externalized.
    
    Parameters
    ----------
    data_set: DataSet
        The data set to check
        
    Returns
    -------
    map of constraing number to errors
        A map containing a list of error descriptions for each constraint number listed above
    """
    log = logging.getLogger(__name__)

    violations = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: []
    }

    # check constraints 1-4, 6, and 7
    group_by_filename = data_set._data.groupby(_DataVar.FILENAME)

    num_chunks = None

    for filename, filename_group in group_by_filename:  # type: xr.Dataset
        num_violations = 0

        # constraint 1: same nominal labels for same filename
        labels_nominal = list(np.unique(filename_group[_DataVar.LABEL_NOMINAL].fillna(_MISSING_VALUE_STR).values))

        if len(labels_nominal) != 1:
            violations[0].append("instances for file {} have different nominal labels: {}"
                                 .format(filename, _format_values(labels_nominal, _MISSING_VALUE_STR)))
            num_violations += 1

        # constraint 2: same numeric label for same filename
        labels_numeric = list(np.unique(filename_group[_DataVar.LABEL_NUMERIC].fillna(_MISSING_VALUE_INT).values))

        if len(labels_numeric) != 1:
            violations[1].append("instances for file {} have different numeric labels: {}"
                                 .format(filename, _format_values(labels_numeric, _MISSING_VALUE_INT)))
            num_violations += 1

        # constraint 3: same cross validation info for same filename
        for fold in range(filename_group.dims[_Dimension.FOLD]):
            splits = list(np.unique(filename_group[{_Dimension.FOLD: fold}][_DataVar.CV_FOLDS]
                                    .fillna(_MISSING_VALUE_INT).values))

            if len(splits) != 1:
                violations[2].append("instances for file {} have different splits in cross validation fold {}: {}"
                                     .format(filename, fold, _format_enum_values(splits, Split)))
                num_violations += 1

        # constraint 4: same partition information for same filename
        partitions = list(np.unique(filename_group[_DataVar.PARTITION].fillna(_MISSING_VALUE_INT).values))

        if len(partitions) != 1:
            violations[3].append("instances for file {} have different partitions: {}"
                                 .format(filename, _format_enum_values(partitions, Partition)))
            num_violations += 1

        # constraint 6: same number of chunks for each filename
        if num_chunks is None:
            num_chunks = filename_group.dims[_Dimension.INSTANCE]
        elif num_chunks != filename_group.dims[_Dimension.INSTANCE]:
            violations[5].append("file {} has invalid number of chunks: expected {}, got {}"
                                 .format(filename, num_chunks, filename_group.dims[_Dimension.INSTANCE]))
            num_violations += 1

        # constraint 7: correct chunk indices for each filename
        sorted_filename_group = filename_group.fillna(_MISSING_VALUE_INT).sortby(_DataVar.CHUNK_NR)

        for index in range(sorted_filename_group.dims[_Dimension.INSTANCE]):
            chunk_nr = sorted_filename_group[{_Dimension.INSTANCE: index}][_DataVar.CHUNK_NR].values

            if chunk_nr != index:
                violations[6].append("invalid chunk number for file {}: expected {}, got {}"
                                     .format(filename, index, "None" if chunk_nr == _MISSING_VALUE_INT else chunk_nr))
                num_violations += 1

        log.debug("processed filename %s -> %d violations", filename, num_violations)

    # constraint 5: if a label map is given, all nominal and numeric labels must match the mapping
    if data_set.label_map is not None:
        log.debug("processing labels")

        for index in data_set:
            instance = data_set[index]

            label_nominal = instance.label_nominal
            label_numeric = instance.label_numeric

            if (label_nominal is None) ^ (label_numeric is None) \
                    or (label_nominal is not None and label_nominal not in data_set.label_map) \
                    or (label_nominal is not None and label_numeric != data_set.label_map[label_nominal]):
                violations[4].append("chunk {} of file {} has invalid labels: {} ({})"
                                     .format(instance.chunk_nr, instance.filename, label_nominal, label_numeric))

    return None if all([not x for x in violations.values()]) else violations
