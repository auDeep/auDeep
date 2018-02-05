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

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cliff.command import Command
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from audeep.backend.data.data_set import load, DataSet
from audeep.backend.log import LoggingMixin


class VisualizeTSNE(LoggingMixin, Command):
    """
    Compute and plot a T-SNE embedding of a data set.
    """

    def __init__(self, app, app_args):
        super().__init__(app, app_args)

    def get_parser(self, prog_name):
        parser = super(VisualizeTSNE, self).get_parser(prog_name)

        parser.add_argument("--input",
                            type=Path,
                            required=True,
                            help="File containing a data set in netCDF 4 format, conformant to the auDeep data "
                                 "model")
        parser.add_argument("--perplexity",
                            type=int,
                            default=50,
                            help="The perplexity parameter for T-SNE (default 50)")
        parser.add_argument("--learning-rate",
                            type=int,
                            default=1000,
                            help="The learning rate for T-SNE (default 1000)")
        return parser

    def plot_with_labels(self,
                         data_set: DataSet,
                         embedding: np.ndarray):
        assert embedding.shape[0] >= len(data_set.label_map), "More labels than weights"

        # use pandas to get indices of instances with the same label
        df = pd.DataFrame({"labels_numeric": data_set.labels_numeric})
        label_indices = {label: indices.tolist() for label, indices in df.groupby(df.labels_numeric).groups.items()}

        norm = matplotlib.colors.Normalize(vmin=min(label_indices.keys()), vmax=max(label_indices.keys()), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap="jet")

        for label, indices in label_indices.items():
            coords = embedding[indices, :]
            plt.plot(coords[:, 0], coords[:, 1], 'o', ms=6, color=mapper.to_rgba(label))

        handles = []

        # noinspection PyTypeChecker
        for key in data_set.label_map:
            handles.append(matplotlib.patches.Patch(color=mapper.to_rgba(data_set.label_map[key]), label=key))

        plt.legend(handles=handles,
                   bbox_to_anchor=(0, -0.125, 1, 0),
                   loc=2,
                   mode="expand",
                   borderaxespad=0.,
                   ncol=3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.33)
        plt.show()

    def take_action(self, parsed_args):
        if not parsed_args.input.exists():
            raise IOError("failed to open data set at {}".format(parsed_args.input))

        data_set = load(parsed_args.input)

        features = np.reshape(data_set.features, [data_set.num_instances, -1])

        if features.shape[1] > 50:
            self.log.info("applying PCA")

            pca = PCA(n_components=200)
            pca.fit(features)
            features = pca.transform(features)

        self.log.info("computing T-SNE embedding")
        tsne = TSNE(perplexity=parsed_args.perplexity,
                    learning_rate=parsed_args.learning_rate,
                    verbose=self.app_args.verbose_level)

        embedding = tsne.fit_transform(features)

        self.log.info("plotting embedding")
        self.plot_with_labels(data_set, embedding)
