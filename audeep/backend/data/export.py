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

"""Data set export to TFRecords, CSV, or ARFF"""
import logging
from enum import Enum
from pathlib import Path

# noinspection PyPackageRequirements
import arff
import numpy as np
import pandas as pd
import tensorflow as tf

from audeep.backend.data import records
# noinspection PyProtectedMember
from audeep.backend.data.data_set import DataSet, Partition, _DataVar, Split


class ExportFormat(Enum):
    """
    Enum for supported data set export formats.
    """
    CSV = 0
    ARFF = 1


def export_tfrecords(path: Path,
                     data_set: DataSet):
    """
    Export the feature matrices of the specified data set in TFRecords format.
    
    No metadata at all is written, so that no label information can be accidentally used during training of a feature
    learning system. Any directories in the path that do not exist are automatically created.
    
    Parameters
    ----------
    path: pathlib.Path
        The output file. 
    data_set: DataSet
        The data set which should be exported
    """
    log = logging.getLogger(__name__)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    log.info("writing data set as TFRecords to %s", path)

    writer = tf.python_io.TFRecordWriter(str(path))

    for index in data_set:
        instance = data_set[index]

        writer.write(records.to_example(instance.features).SerializeToString())

    writer.close()


# noinspection PyProtectedMember
def _write_csv(outfile: Path,
               data_set: DataSet,
               labels_last: bool):
    """
    Write the specified data set to the specified path in CSV format.
    
    This function exports the filename, chunk number, nominal label, and numeric label of instances as metadata columns,
    in addition to the flattened feature matrix of each instance. The filename and chunk number are always written as
    the first two columns. Depending on the choice of the `labels_last` parameter, the labels are written as the next
    two columns, or appended to the end of the columns. Feature columns are given a generic name.
    
    Parameters
    ----------
    outfile: pathlib.Path
        The output path
    data_set: DataSet
        The data set which should be exported
    labels_last: bool
        If set, write the labels as the last two columns. Otherwise, write them as the third and fourth column after the
        filename and chunk number
    """
    # convert to pandas DataFrame
    features_flattened = data_set.features.reshape(data_set.num_instances, -1)
    feature_names = ["feature_%d" % index for index in range(features_flattened.shape[-1])]

    data_frame = pd.DataFrame(data=features_flattened,
                              columns=feature_names)

    xr_data = data_set._data

    data_frame.insert(0, _DataVar.FILENAME, xr_data[_DataVar.FILENAME].values)
    data_frame.insert(1, _DataVar.CHUNK_NR, xr_data[_DataVar.CHUNK_NR].values)

    if labels_last:
        data_frame.insert(data_frame.shape[1], _DataVar.LABEL_NOMINAL, xr_data[_DataVar.LABEL_NOMINAL].values)
        data_frame.insert(data_frame.shape[1], _DataVar.LABEL_NUMERIC, xr_data[_DataVar.LABEL_NUMERIC].values)
    else:
        data_frame.insert(2, _DataVar.LABEL_NOMINAL, xr_data[_DataVar.LABEL_NOMINAL].values)
        data_frame.insert(3, _DataVar.LABEL_NUMERIC, xr_data[_DataVar.LABEL_NUMERIC].values)

    # filename might contain dots
    filename = str(outfile)

    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    data_frame.to_csv(filename, index=False)


def _write_arff(outfile: Path,
                data_set: DataSet,
                labels_last: bool):
    """    
    Write the specified data set to the specified path in ARFF format.
    
    This function exports the filename, chunk number, nominal label, and numeric label of instances as metadata 
    attributes, in addition to the flattened feature matrix of each instance. The filename and chunk number are 
    always written as the first two attributes. Depending on the choice of the `labels_last` parameter, the labels are 
    written as the next two attributes, or appended to the end of the attributes. Feature attributes are given a 
    generic name.
    
    Parameters
    ----------
    outfile: pathlib.Path
        The output path
    data_set: DataSet
        The data set which should be exported
    labels_last: bool
        If set, write the labels as the last two attributes. Otherwise, write them as the third and fourth attributes 
        after the filename and chunk number
    """
    if data_set.label_map is not None:
        nominal_labels = sorted(data_set.label_map.keys())
    else:
        # noinspection PyProtectedMember
        nominal_labels = sorted(np.unique(data_set._data[_DataVar.LABEL_NOMINAL].values).tolist())

    data_arff = {
        "relation": "auDeep exported data set",
        "attributes": [
            (_DataVar.FILENAME, "STRING"),
            (_DataVar.CHUNK_NR, "INTEGER"),
        ],
        "data": []
    }

    num_features = 1

    for dim in data_set.feature_shape:
        num_features *= dim

    if not labels_last:
        # noinspection PyTypeChecker
        data_arff["attributes"].append((_DataVar.LABEL_NOMINAL, nominal_labels)),
        data_arff["attributes"].append((_DataVar.LABEL_NUMERIC, "INTEGER"))

    for feature in range(num_features):
        data_arff["attributes"].append(("feature_%d" % feature, "REAL"))

    if labels_last:
        # noinspection PyTypeChecker
        data_arff["attributes"].append((_DataVar.LABEL_NOMINAL, nominal_labels)),
        data_arff["attributes"].append((_DataVar.LABEL_NUMERIC, "INTEGER"))

    for index in data_set:
        instance = data_set[index]
        row_labels = [instance.label_nominal, instance.label_numeric]

        row = [instance.filename, instance.chunk_nr]

        if not labels_last:
            row += row_labels

        row += instance.features.flatten().tolist()

        if labels_last:
            row += row_labels

        data_arff["data"].append(row)

    # filename might contain dots
    filename = str(outfile)

    if not filename.endswith(".arff"):
        filename = filename + ".arff"

    with open(filename, "w", newline="\n") as fp:
        arff.dump(data_arff, fp)


def _write(outfile: Path,
           data_set: DataSet,
           labels_last: bool,
           fmt: ExportFormat):
    """
    Write the specified data set to the specified path.
    
    The output format is determined based on the value of the `fmt` parameter.
    
    Parameters
    ----------
    outfile: pathlib.Path
        The output path
    data_set: DataSet
        The data set which should be exported
    labels_last: bool
        If set, write the labels as the last two columns/attributes. Otherwise, write them as the third and fourth 
        columns/attributes after the filename and chunk number
    fmt: ExportFormat
        The output format
        
    Raises
    ------
    ValueError
        If the output format is unknown
    """
    if fmt is ExportFormat.CSV:
        _write_csv(outfile, data_set, labels_last)
    elif fmt is ExportFormat.ARFF:
        _write_arff(outfile, data_set, labels_last)
    else:
        raise ValueError("unknown format: {}".format(fmt))


def export(basedir: Path,
           name: str,
           data_set: DataSet,
           labels_last: bool,
           fmt: ExportFormat):
    """
    Export the specified data set.
    
    The data set is written in several files distributed over a certain directory structure below the specified base
    directory, depending on whether partition or cross-validation information is present.
    
    If the data set has neither partition nor cross-validation information, it is written to a single file directly 
    below the specified base directory.
    
    If the data set has only partition information, a folder is created below the base directory for each partition, 
    and the partitions are written separately to a single file in the respective partition directory.
    
    If the data set has only cross-validation information, a folder called `fold_N` is created for each cross-validation
    fold `N`, and the validation split of each fold is written to a single file in the respective fold directory. Please
    note that this directory structure can not accurately represent data sets with overlapping validation splits, in
    which case some instances will be duplicated.
    
    If the data set has both partition and cross-validation information, the above two strategies are combined, by first
    creating a directory for each partition, and then creating fold directories below each partition directory.
    
    The filename of files written by this function can be set using the parameter `name`, and the extension is chosen
    depending on the choice of output format. Any directories in the base directory path that do not exist will be
    created automatically.
    
    Parameters
    ----------
    basedir: pathlib.Path
        The output base directory
    name: str
        The output file name
    data_set: DataSet
        The data set to export
    labels_last: bool
        If set, write the labels as the last two columns/attributes. Otherwise, write them as the third and fourth 
        columns/attributes after the filename and chunk number
    fmt: ExportFormat
        The output format
    """
    log = logging.getLogger(__name__)

    if not basedir.exists():
        basedir.mkdir(parents=True)

    if len(data_set.feature_shape) > 1:
        log.warning("data set has more than one feature dimension - features will be flattened")

    if not data_set.has_partition_info and not data_set.has_cv_info:
        # data set has neither partition info nor cross validation info
        _write_csv(outfile=basedir / name,
                   data_set=data_set,
                   labels_last=labels_last)
    elif not data_set.has_partition_info:
        # data set has only cv info
        if data_set.has_overlapping_folds:
            log.warning("data set has overlapping cross validation folds - some instances will be duplicated")

        for fold in range(data_set.num_folds):
            fold_dir = basedir / ("fold_%d" % (fold + 1))

            if not fold_dir.exists():
                fold_dir.mkdir()

            log.info("writing fold %d to %s.%s", fold + 1, fold_dir / name, fmt.name.lower())
            _write(outfile=fold_dir / name,
                   data_set=data_set.split(fold, Split.VALID),
                   labels_last=labels_last,
                   fmt=fmt)
    elif not data_set.has_cv_info:
        # data set has only partition info
        for partition in Partition:
            partition_data_set = data_set.partitions(partition)

            if partition_data_set.num_instances > 0:
                partition_dir = basedir / partition.name.lower()

                if not partition_dir.exists():
                    partition_dir.mkdir()

                log.info("writing partition %s to %s.%s", partition.name.lower(), partition_dir / name,
                         fmt.name.lower())

                _write(outfile=partition_dir / name,
                       data_set=partition_data_set,
                       labels_last=labels_last,
                       fmt=fmt)
    else:
        # data set has partition and cv info
        for partition in Partition:
            partition_data_set = data_set.partitions(partition)

            if partition_data_set.num_instances > 0:
                partition_dir = basedir / partition.name.lower()

                if not partition_dir.exists():
                    partition_dir.mkdir()

                if partition_data_set.has_overlapping_folds:
                    log.warning("partition %s of data set has overlapping cross validation folds - some instances will "
                                "be duplicated", partition.name.lower())

                for fold in range(partition_data_set.num_folds):
                    fold_dir = partition_dir / ("fold_%d" % (fold + 1))

                    if not fold_dir.exists():
                        fold_dir.mkdir()

                    log.info("writing partition %s fold %d to %s.%s", partition.name.lower(), fold + 1, fold_dir / name,
                             fmt.name.lower())
                    _write(outfile=fold_dir / name,
                           data_set=data_set.split(fold, Split.VALID),
                           labels_last=labels_last,
                           fmt=fmt)


def export_csv(basedir: Path,
               name: str,
               data_set: DataSet,
               labels_last: bool = False):
    """
    Shorthand function for CSV export.
    
    Delegates to the `export` function.
    
    Parameters
    ----------
    basedir: pathlib.Path
        The output base directory
    name: str
        The output file name
    data_set: DataSet
        The data set to export
    labels_last: bool
        If set, write the labels as the last two columns/attributes. Otherwise, write them as the third and fourth 
        columns/attributes after the filename and chunk number
    """
    export(basedir=basedir,
           name=name,
           data_set=data_set,
           labels_last=labels_last,
           fmt=ExportFormat.CSV)


def export_arff(basedir: Path,
                name: str,
                data_set: DataSet,
                labels_last: bool = False):
    """
    Shorthand function for ARFF export.
    
    Delegates to the `export` function.
    
    Parameters
    ----------
    basedir: pathlib.Path
        The output base directory
    name: str
        The output file name
    data_set: DataSet
        The data set to export
    labels_last: bool
        If set, write the labels as the last two columns/attributes. Otherwise, write them as the third and fourth 
        columns/attributes after the filename and chunk number
    """
    export(basedir=basedir,
           name=name,
           data_set=data_set,
           labels_last=labels_last,
           fmt=ExportFormat.ARFF)
