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

"""Data set import from CSV/ARFF"""
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, List

import arff
import numpy as np
import pandas as pd

from audeep.backend.data.data_set import DataSet, _DataVar, Partition, empty, concat_instances, Split
from audeep.backend.log import LoggingMixin


def _is_known_suffix(suffix: str) -> bool:
    """
    Check whether we can parse the file type indicated by the specified file extension.
    
    Currently, we can parse CSV and ARFF data.
    
    Parameters
    ----------
    suffix: str
        The file extension, with a leading dot

    Returns
    -------
    bool
        True, if we can parse the file type indicated by the specified file extension
    """
    return suffix.lower() == ".csv" or suffix.lower() == ".arff"


def _build_label_map(nominal_labels: Iterable[str],
                     numeric_labels: Iterable[int]) -> Mapping[str, int]:
    """
    Build a mapping of nominal to numeric labels from the specified labels.
    
    The lists passed to this function should contain the nominal and numeric labels of instances in order, that is,
    `nominal_labels[i]` should contain the nominal label of the i-th instance, and `numeric_labels[i]` should contain
    the numeric label of the i-th instance.
    
    This function iterates over all instances and checks for consistency of the label mapping. 
    
    Parameters
    ----------
    nominal_labels: list of str
        The nominal labels of instances
    numeric_labels: list of int
        The numeric labels of instances

    Returns
    -------
    map of str to int
        A mapping of nominal labels to numeric labels consistent with the labels passed to this function
        
    Raises
    ------
    IOError
        If the label assignments passed to this function are inconsistent, i.e. if there are multiple numeric labels
        for one nominal label or vice-versa
    """
    label_map = {}
    label_map_inverse = {}

    for nominal_label, numeric_label in zip(nominal_labels, numeric_labels):
        if nominal_label in label_map and label_map[nominal_label] != numeric_label:
            raise IOError("error while importing data set: nominal label %s has multiple numeric values: "
                          "%d, %d" % (nominal_label, numeric_label, label_map[nominal_label]))
        elif numeric_label in label_map_inverse and label_map_inverse[numeric_label] != nominal_label:
            raise IOError("error while importing data set: numeric label %s has multiple nominal values: "
                          "%s, %d" % (numeric_label, nominal_label, label_map_inverse[numeric_label]))
        else:
            label_map[nominal_label] = numeric_label
            label_map_inverse[numeric_label] = nominal_label

    return label_map


class DataImporter(LoggingMixin):
    """
    Helper class for importing data sets from CSV or ARFF.
    """

    def __init__(self,
                 filename_attribute: str = _DataVar.FILENAME,
                 chunk_nr_attribute: str = _DataVar.CHUNK_NR,
                 label_nominal_attribute: str = _DataVar.LABEL_NOMINAL,
                 label_numeric_attribute: str = _DataVar.LABEL_NUMERIC):
        """
        Create and initialize a new DataImporter with the specified parameters.
        
        The parameters can be used to customize the names of metadata columns/attributes in the data which should be
        imported. They default to the names used internally in the DataSet class.
        
        Parameters
        ----------
        filename_attribute: str
            The name of the filename attribute/column
        chunk_nr_attribute: str
            The name of the chunk number attribute/column
        label_nominal_attribute: str
            The name of the nominal label attribute/column
        label_numeric_attribute: str
            The name of the numeric label attribute/column
        """
        super().__init__()

        self._filename_attribute = filename_attribute
        self._chunk_nr_attribute = chunk_nr_attribute,
        self._label_nominal_attribute = label_nominal_attribute
        self._label_numeric_attribute = label_numeric_attribute

    def _import_arff(self,
                     file: Path,
                     num_folds: int,
                     fold_index: Optional[int],
                     partition: Optional[Partition]) -> DataSet:
        """
        Import a data set from ARFF.
        
        Besides feature attributes, the ARFF file must at least contain a nominal label attribute. If additionally a
        numeric label attribute is present, a label map is generated from the nominal and numeric labels. Otherwise, a
        label map is synthesized. If no filename attribute is present, synthetic filenames are used. However, this will
        trigger a warning, since most of the data set integrity checks rely on the filenames being known.
        
        Any attributes that are not recognized as metadata attributes are assumed to contain numeric features.
        
        Parameters
        ----------
        file: pathlib.Path
            The file from which to import the data set
        num_folds: int
            The number of folds to create in the data set
        fold_index: int, optional
            The fold to which the instances in the data set belong. Ignored if `num_folds` is zero
        partition: Partition, optional
            The partition to which the instances in the data set belong

        Returns
        -------
        DataSet
            A data set containing instances imported from the specified ARFF file
            
        Raises
        ------
        IOError
            If an error occurs while importing the data set
        """
        with open(str(file)) as fp:
            data_arff = arff.load(fp)

        attribute_names, attribute_types = list(zip(*data_arff["attributes"]))
        data = data_arff["data"]

        if self._label_nominal_attribute not in attribute_names:
            raise IOError("error while importing data set from %s: required nominal label column %s missing" %
                          (file, self._label_nominal_attribute))

        label_nominal_index = attribute_names.index(self._label_nominal_attribute)

        filename_exists = self._filename_attribute in attribute_names
        chunk_nr_exists = self._chunk_nr_attribute in attribute_names
        label_numeric_exists = self._label_numeric_attribute in attribute_names

        num_instances = len(data)

        metadata_columns = [label_nominal_index]

        if filename_exists:
            filename_index = attribute_names.index(self._filename_attribute)
            metadata_columns.append(filename_index)
        else:
            self.log.warning("no filename attribute found, validation of data set integrity will be impossible")

        if chunk_nr_exists:
            chunk_nr_index = attribute_names.index(self._chunk_nr_attribute)
            metadata_columns.append(chunk_nr_index)

        if label_numeric_exists:
            label_numeric_index = attribute_names.index(self._label_numeric_attribute)

            metadata_columns.append(label_numeric_index)

            nominal_labels = [row[label_nominal_index] for row in data]
            numeric_labels = [row[label_numeric_index] for row in data]

            label_map = _build_label_map(nominal_labels, numeric_labels)
        else:
            nominal_labels = np.unique([row[label_nominal_index] for row in data]).tolist()

            # noinspection PyTypeChecker
            label_map = dict(zip(nominal_labels, range(len(nominal_labels))))

        num_features = len(data[0]) - len(metadata_columns)

        result = empty(num_instances=num_instances,
                       feature_dimensions=[("generated", num_features)],
                       num_folds=num_folds)
        result.label_map = label_map

        if num_folds > 0:
            cv_folds = [Split.TRAIN] * num_folds
            cv_folds[fold_index] = Split.VALID
        else:
            cv_folds = None

        for index, row in enumerate(data):
            instance = result[index]

            # noinspection PyUnboundLocalVariable
            instance.filename = "synthetic_%s_%d" % (file, index) if not filename_exists else row[filename_index]
            # noinspection PyUnboundLocalVariable
            instance.chunk_nr = 0 if not chunk_nr_exists else row[chunk_nr_index]
            instance.label_nominal = row[label_nominal_index]
            instance.cv_folds = cv_folds
            instance.partition = partition
            instance.features = np.array([row[i] for i in set(range(len(row))).difference(set(metadata_columns))])

            self.log.debug("read instance %s (%d/%d)", instance.filename, index + 1, num_instances)

        return result

    def _import_csv(self,
                    file: Path,
                    num_folds: int,
                    fold_index: Optional[int],
                    partition: Optional[Partition]) -> DataSet:
        """
        Import a data set from CSV.
        
        Besides feature columns, the CSV file must at least contain a nominal label columns. If additionally a
        numeric label columns is present, a label map is generated from the nominal and numeric labels. Otherwise, a
        label map is synthesized. If no filename columns is present, synthetic filenames are used. However, this will
        trigger a warning, since most of the data set integrity checks rely on the filenames being known.
        
        Any columns that are not recognized as metadata attributes are assumed to contain numeric features.
        
        Parameters
        ----------
        file: pathlib.Path
            The file from which to import the data set
        num_folds: int
            The number of folds to create in the data set
        fold_index: int, optional
            The fold to which the instances in the data set belong. Ignored if `num_folds` is zero
        partition: Partition, optional
            The partition to which the instances in the data set belong

        Returns
        -------
        DataSet
            A data set containing instances imported from the specified CSV file
            
        Raises
        ------
        IOError
            If an error occurs while importing the data set
        """
        data_frame = pd.read_csv(file)  # type: pd.DataFrame

        if self._label_nominal_attribute not in data_frame:
            raise IOError("error while importing data set from %s: required nominal label column %s missing" %
                          (file, self._label_nominal_attribute))

        filename_exists = self._filename_attribute in data_frame
        chunk_nr_exists = self._chunk_nr_attribute in data_frame
        label_numeric_exists = self._label_numeric_attribute in data_frame

        num_instances = len(data_frame)

        metadata_columns = [self._label_nominal_attribute]

        if filename_exists:
            metadata_columns.append(self._filename_attribute)
        else:
            self.log.warning("no filename attribute found, validation of data set integrity will be impossible")

        if chunk_nr_exists:
            metadata_columns.append(self._chunk_nr_attribute)

        if label_numeric_exists:
            metadata_columns.append(self._label_numeric_attribute)

            nominal_labels = data_frame[self._label_nominal_attribute]
            numeric_labels = data_frame[self._label_numeric_attribute]

            label_map = _build_label_map(nominal_labels, numeric_labels)
        else:
            nominal_labels = np.unique(data_frame[self._label_nominal_attribute]).tolist()

            # noinspection PyTypeChecker
            label_map = dict(zip(nominal_labels, range(len(nominal_labels))))

        num_features = len(data_frame.columns) - len(metadata_columns)

        result = empty(num_instances=num_instances,
                       feature_dimensions=[("generated", num_features)],
                       num_folds=num_folds)
        result.label_map = label_map

        if num_folds > 0:
            cv_folds = [Split.TRAIN] * num_folds
            cv_folds[fold_index] = Split.VALID
        else:
            cv_folds = None

        for index in result:
            data_frame_row = data_frame.iloc[index]  # type: pd.Series
            instance = result[index]

            instance.filename = "synthetic_%s_%d" % (file, index) if not filename_exists else data_frame_row[
                self._filename_attribute]
            instance.chunk_nr = 0 if not chunk_nr_exists else data_frame_row[self._chunk_nr_attribute]
            instance.label_nominal = data_frame_row[self._label_nominal_attribute]
            instance.cv_folds = cv_folds
            instance.partition = partition
            # noinspection PyUnresolvedReferences
            instance.features = data_frame_row.drop(metadata_columns).values.astype(np.float32)

            self.log.debug("read instance %s (%d/%d)", instance.filename, index + 1, num_instances)

        return result

    def _import(self,
                file: Path,
                num_folds: int = 0,
                fold_index: Optional[int] = None,
                partition: Optional[Partition] = None) -> DataSet:
        """
        Import a data set from CSV or ARFF.
        
        This method decides based on the file extension which parser to use.
        
        Parameters
        ----------
        file: pathlib.Path
            The file from which to import the data set
        num_folds: int
            The number of folds to create in the data set
        fold_index: int, optional
            The fold to which the instances in the data set belong. Ignored if `num_folds` is zero
        partition: Partition, optional
            The partition to which the instances in the data set belong

        Returns
        -------
        DataSet
            A data set containing instances imported from the specified file
            
        Raises
        ------
        IOError
            If the file extension is unknown
        """
        if file.suffix.lower() == ".csv":
            return self._import_csv(file=file,
                                    num_folds=num_folds,
                                    fold_index=fold_index,
                                    partition=partition)
        elif file.suffix.lower() == ".arff":
            return self._import_arff(file=file,
                                     num_folds=num_folds,
                                     fold_index=fold_index,
                                     partition=partition)
        else:
            raise IOError("unknown extension: %s" % file.suffix)

    def _import_partitioned(self,
                            partition_dirs: List[Path],
                            name: str) -> List[DataSet]:
        """
        Import data set partitions from the specified directories.
        
        Each directory in `partition_dirs` must be named after a partition, i.e. "train", "devel", or "test". In each
        of the specified directories, exactly one CSV or ARFF file with the specified name is expected, from which 
        instances for the respective partition are imported.
        
        Parameters
        ----------
        partition_dirs: list of pathlib.Path
            Partition directories named "train", "devel", or "test"
        name: str
            The name of the files from which to import instances

        Returns
        -------
        list of DataSet
            A list containing one DataSet for each partition that was imported
            
        Raises
        ------
        IOError
            If an error occurs while importing the data sets
        """
        self.log.debug("trying to import partitioned data set")

        partition_dir_map = dict([(Partition[dir.name.upper()], dir) for dir in partition_dirs])
        partition_file_map = {}

        # scan partition directories, and check for each partition if there is a data file that we can parse
        for partition in partition_dir_map:
            files = []

            for file in partition_dir_map[partition].glob(name + ".*"):
                if _is_known_suffix(file.suffix):
                    self.log.debug("found data file for partition %s: %s" % (partition.name, file))

                    files.append(file)

            if len(files) > 1:
                raise IOError("error while importing partitioned data set: ambiguous data files for partition %s: %s"
                              % (partition.name, files))

            if len(files) == 1:
                partition_file_map[partition] = files[0]

        if len(partition_file_map) == 0:
            raise IOError("error while importing partitioned data set: no data files found")

        self.log.debug("importing data set with %d partitions" % len(partition_file_map))

        data_sets = []

        for partition in partition_file_map:
            self.log.debug("reading partition %s from data file %s" % (partition.name, partition_file_map[partition]))

            data_sets.append(self._import(file=partition_file_map[partition],
                                          partition=partition))

        return data_sets

    def _import_cross_validated(self,
                                fold_dirs: List[Path],
                                name: str) -> List[DataSet]:
        """
        Import data set cross-validation folds from the specified directories.

        Each directory in `fold_dirs` must be named "fold_N", where `N` indicates the fold number, and fold directories
        must be consecutively numbered from one to the number of folds. In each of the specified directories, exactly 
        one CSV or ARFF file with the specified name is expected, from which instances for the respective fold are 
        imported.

        Parameters
        ----------
        fold_dirs: list of pathlib.Path
            Cross-validation fold directories named "fold_N"
        name: str
            The name of the files from which to import instances

        Returns
        -------
        list of DataSet
            A list containing one DataSet for each cross-validation fold that was imported

        Raises
        ------
        IOError
            If an error occurs while importing the data sets
        """
        self.log.debug("trying to import cross-validated data set")

        num_folds = len(fold_dirs)

        if num_folds < 2:
            raise IOError("error while importing cross-validated data set: expected two or more fold directories, "
                          "got %s" % num_folds)

        fold_files = []

        for fold_index, fold_dir in enumerate(sorted(fold_dirs, key=lambda f: f.name)):
            fold_number = int(fold_dir.name[5:])

            if fold_index + 1 != fold_number:
                raise IOError("error while importing cross-validated data set: directory for fold %d missing"
                              % (fold_index + 1))

            files = []

            for file in fold_dir.glob(name + ".*"):
                if _is_known_suffix(file.suffix):
                    self.log.debug("found data file for fold %d: %s" % (fold_number, file))

                    files.append(file)

            if len(files) > 1:
                raise IOError("error while importing cross-validated data set: ambiguous data files for fold %d: %s"
                              % (fold_number, files))

            if len(files) == 1:
                fold_files.append(files[0])

        data_sets = []

        for fold_index, fold_file in enumerate(fold_files):
            self.log.debug("reading fold %d from data file %s" % (fold_index + 1, fold_file))

            data_sets.append(self._import(file=fold_file,
                                          num_folds=num_folds,
                                          fold_index=fold_index))

        return data_sets

    def import_data(self,
                    basedir: Path,
                    name: str) -> DataSet:
        """
        Import a data set with the specified name from the specified base directory.
        
        The base directory may contain a single CSV or ARFF file with the specified name, in which case a data set
        without partition or cross-validation information is imported.
        
        In order to import a data set with partition information, there must be one CSV or ARFF file with the specified
        name for each partition, in a subfolder named after the respective partition.
        
        In order to import a data set with cross-validation information, there must be one CSV or ARFF file with the
        specified name for each cross-validation fold, in subfolders named "fold_N" where `N` indicates the fold number.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The base directory from which to import a data set
        name: str
            The name of data files from which to import instances

        Returns
        -------
        DataSet
            A data set containing the imported instances and metadata
            
        Raises
        ------
        IOError
            If an error occurs while importing the data set, either due to OS errors or due to incorrect file structure
        """
        # check if we are importing a single file
        single_files = list(basedir.glob(name + ".*"))

        if len(single_files) == 1:
            data_file = single_files[0]

            self.log.debug("trying to import data set without partition or cross validation information from %s",
                           data_file)

            return self._import(file=data_file)
        elif len(single_files) > 1:
            raise IOError("error while importing data set: ambiguous data files %s" % single_files)

        dirs = [file for file in basedir.glob("*") if file.is_dir()]

        # check if we are importing partitioned data
        if all([dir.name in ["train", "devel", "test"] for dir in dirs]):
            data_sets = self._import_partitioned(partition_dirs=dirs,
                                                 name=name)

            result = concat_instances(data_sets)
            result.freeze()

            return result

        # check if we are importing cross-validated data
        fold_pattern = re.compile("fold_[\d]+")

        if all([fold_pattern.match(dir.name) for dir in dirs]):
            data_sets = self._import_cross_validated(fold_dirs=dirs,
                                                     name=name)

            result = concat_instances(data_sets)
            result.freeze()

            return result

        raise IOError("cannot import data set from %s: base directory must either contain up to three partition "
                      "directories or two or more fold directories" % basedir)
