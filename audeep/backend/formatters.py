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

"""Different output formatters for printing data to the console"""
import math
import re
from typing import Sequence, Mapping, List

import numpy as np


class TableFormatter:
    """
    A formatter for printing tabular data to the standard output stream.
    """

    def __init__(self,
                 alignment: str = None):
        """
        Create and initialize a new TableFormatter with the specified alignment.
        
        The alignment may be omitted, in which case all columns will be left-aligned. Each character of the alignment
        string specifies whether the corresponding column should be left-aligned ('l'), or right-aligned ('r'). If more
        columns are passed to the `print` method than there are characters in the alignment string, the surplus columns
        will be left-aligned.
        
        Parameters
        ----------
        alignment: str, optional
            Alignment string specifying the alignment of columns
        """
        if alignment is not None and not re.compile("[lr]*").match(alignment):
            raise ValueError("alignment string must only contain 'l' and 'r' characters")

        self._alignment = alignment

    def _get_column_widths(self, data: Sequence[Sequence[str]]) -> List[int]:
        """
        Returns the maximum widths for each individual column in the specified tabular data.
        
        Parameters
        ----------
        data

        Returns
        -------

        """
        # little bit of functional programming fun:
        #
        # - [map(len, row) for row in data] is a list containing a list of columns widths for each row
        #
        # - zip(*[map(len, row) for row in data]) transposes that list, so that it contains one list for each column
        # containing the widths of the column in each row
        #
        # map(max, column_widths) finally returns the maximum width of each column
        column_widths = list(zip(*[map(len, row) for row in data]))

        # noinspection PyTypeChecker
        return list(map(max, column_widths))

    def _print_divider(self,
                       column_widths: Sequence[int]):
        """
        Prints a horizontal divider.
        
        Parameters
        ----------
        column_widths: list of int
            The widths of columns
        """
        print("+", end="")
        for i in range(len(column_widths)):
            print("%s+" % ("-" * (column_widths[i] + 2)), end="")
        print()

    def print(self,
              data: Sequence[Sequence[str]],
              header: str = None,
              dividers: Sequence[int] = None):
        """
        Prints the specified tabular data to standard output.
        
        Optionally, a table header may be specified using the `header` parameter. The `dividers` parameter can be used
        to specify additional horizontal dividers. Each entry in the list specifies a row index after which a divider
        should be printed.
        
        Parameters
        ----------
        data: list of list of str
            Data to print. Must be a nested list, where the outer list corresponds rows, and the inner lists correspond
            to columns within a row.
        header: str, optional
            A table header
        dividers: list of int, optional
            Additional horizontal dividers
        """
        data = [list(map(lambda x: str(x).strip(), row)) for row in data]

        num_columns = max(len(row) for row in data)

        column_widths = self._get_column_widths(data)

        body_width = sum(column_widths) + 4 + 3 * (num_columns - 1)
        header_width = body_width

        if header is not None:
            header_width = max(body_width, len(header) + 4)

            print("+%s+" % ("-" * (header_width - 2)))
            print("| %%-%ds |" % (header_width - 4) % header)

        if header_width > body_width:
            diff = header_width - body_width

            index = 0

            for _ in range(diff):
                column_widths[index] += 1
                index = (index + 1) % num_columns

        self._print_divider(column_widths)

        for index, row in enumerate(data):
            print("|", end="")

            for i in range(num_columns):
                if self._alignment is not None and len(self._alignment) > i and self._alignment[i] == "r":
                    print(" %%%ds |" % (column_widths[i]) % row[i], end="")
                else:
                    print(" %%-%ds |" % (column_widths[i]) % row[i], end="")

            print()

            if dividers is not None and index in dividers:
                self._print_divider(column_widths)

        self._print_divider(column_widths)


class ConfusionMatrixFormatter:
    """
    Formatter for printing a confusion matrix with labels to stdout.
    """

    def __init__(self,
                 decimals: int = None,
                 normalize: bool = False,
                 abbrev_labels: int = 3):
        """
        Create a new ConfusionMatrixFormatter.
        
        Number of decimals can only be specified if normalization is enabled, otherwise it is ignored.
        
        Parameters
        ----------
        decimals: int, optional
            Number of decimals to print for each confusion matrix entry
        normalize: bool, default False
            Normalize confusion matrix to represent probabilities instead of counts.
        abbrev_labels: int, default 3
            Abbreviate nominal labels to have the specified length
        """
        self._normalize = normalize
        self._abbrev_labels = abbrev_labels

        if normalize:
            if decimals is not None:
                self._decimals = decimals
            else:
                self._decimals = 2

    def format(self,
               confusion_matrix: np.ndarray,
               label_map: Mapping[str, int]) -> str:
        """
        Print the specified confusion matrix using the specified label map.
        
        The `confusion_matrix` parameter must be a square numpy matrix, and the `label_map` parameter must be a mapping
        of nominal to numeric labels, containing one entry for each confusion matrix row/column.
        
        Parameters
        ----------
        confusion_matrix: numpy.ndarray
            A confusion matrix
        label_map: map of str to int
            A label map

        Returns
        -------
        str
            A string representation of the confusion matrix
        """
        if len(confusion_matrix.shape) != 2:
            raise ValueError("confusion matrix must be 2D, is: {}".format(len(confusion_matrix.shape)))
        elif confusion_matrix.shape[0] != confusion_matrix.shape[1]:
            raise ValueError("confusion matrix must be square, is {}".format(confusion_matrix.shape))
        elif confusion_matrix.shape[0] != len(label_map):
            raise ValueError("number of labels must match confusion matrix size, expected: {}, got: {}"
                             .format(confusion_matrix.shape[0], len(label_map)))

        # order nominal labels by numeric values
        ordered_labels = sorted(label_map.items(), key=lambda t: t[1])
        ordered_labels = list(zip(*ordered_labels))[0]

        if self._abbrev_labels is not None:
            ordered_labels = [label[:self._abbrev_labels] if len(label) > self._abbrev_labels else label
                              for label in ordered_labels]

        max_label_len = max(map(len, ordered_labels))

        if self._normalize:
            confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, np.newaxis]

            column_width = max(2 + self._decimals, max_label_len)
            number_pattern = " %%%d.%df" % (column_width, self._decimals)
        else:
            max_digits = int(math.ceil(math.log10(np.max(confusion_matrix))))

            column_width = max(max_digits, max_label_len)
            number_pattern = " %%%dd" % column_width

        result = ""

        label_pattern = "%%%ds" % column_width

        # print header row
        for col in range(confusion_matrix.shape[1] + 1):
            if col == 0:
                result += " " * max_label_len
            else:
                result += " " + label_pattern % ordered_labels[col - 1]

        result += "\n"

        label_pattern = "%%%ds" % max_label_len

        for row in range(confusion_matrix.shape[0]):
            result += label_pattern % ordered_labels[row]

            for col in range(confusion_matrix.shape[1]):
                result += number_pattern % confusion_matrix[row, col]

            result += "\n"

        return result
