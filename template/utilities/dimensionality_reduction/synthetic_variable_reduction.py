# -*- coding: utf-8 -*-
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .tool_utilities.pre_processing import standardize_variables
from .tool_utilities import file_utilities

logger = logging.getLogger(__name__)


def to_excel(result_dict: dict, file_name: str = None, output_path=Path.cwd()):
    timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())

    if not output_path.is_dir():
        os.makedirs(output_path, exist_ok=True)

    if file_name is None:
        file_name = f"PCA_results_{timestamp}.xlsx"

    data = result_dict["data"]
    explained_variance = result_dict["explained_variance"]

    with pd.ExcelWriter(output_path / file_name) as writer:
        data.to_excel(writer, sheet_name='data', index=False)
        explained_variance.to_excel(writer, sheet_name='explained_variance', index=False)


def plot_explained_variance_ratio(result_dict: dict):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Explained variance ratio (cumulative)")
    ax.plot(
        result_dict["plot data"][:, 1],
        result_dict["plot data"][:, 0],
        marker='.',
        markersize=10
    )
    ax.set_xlabel("Component num")
    ax.set_ylabel("Ratio")
    ax.set_ylim([0, 1])
    ax.set_yticks(np.linspace(0, 1, 11))
    plt.grid()

    return fig


class SyntheticVariableReduction:
    @staticmethod
    def pca(
            data, variables, n_components,
            standardize_vars=False,
            generate_charts=False,
            save_results_to_excel=False,
            output_path=Path.cwd() / "outputs"
    ):

        """
        Reduce numerical variables using PCA

        :param data: DataFrame
        :param variables: variable names to be reduced
        :param n_components: maximum number of components tested
        :param generate_charts: Whether to generate charts or not
        :param save_results_to_excel: Whether to save results to excel or not
        :param standardize_vars: Boolean to standardize variables
        :param output_path: Path to save excel and charts

        :return: dictionary, warning_info
        """

        dataset = data[variables]

        if standardize_vars:
            dataset = standardize_variables(dataset)

        # ------------------------ Begin PCA ----------------------------

        print("Starting PCA algorithm")
        pca = PCA(n_components=n_components)
        pca_frame = pd.DataFrame(
            pca.fit_transform(dataset),
            columns=[
                f'pca_dim_{idx}'
                for idx in range(n_components)
            ]
        )

        frame_with_pc = pd.concat([pca_frame, data], axis=1)

        explained_var = pca.explained_variance_ratio_
        explained_var_cum_sum = explained_var.cumsum()
        total_var_explained = explained_var.sum()

        explained_var_cs_df = pd.DataFrame(explained_var_cum_sum)

        list_of_pcs = []
        for x in range(1, n_components+1):
            list_of_pcs.append(f"PC_{x}")

        explained_var_cs_df["PCA_dim"] = list_of_pcs

        # projection of features in the lower space
        hor_dim = len(variables)
        vert_dim = hor_dim
        res = pd.DataFrame(pca.transform(np.eye(hor_dim, vert_dim)), index=list(dataset))

        logger.info('Ending synthetic variable reductions')

        output = dict()
        output["data"] = frame_with_pc
        output["model"] = pca
        output['explained_variance'] = explained_var_cs_df
        output['components'] = res
        output["model_type"] = "Reduction Synthetic"
        output["total_explained_variance"] = total_var_explained
        output["plot data"] = np.array(explained_var_cs_df.values)
        output["plot_type"] = "PCA"

        if save_results_to_excel:
            print("Saving PCA results")
            to_excel(output, output_path=output_path)

        if generate_charts:
            fig = plot_explained_variance_ratio(output)
            file_utilities.export_plot(figure=fig, prefix="explained_variance", output_path=output_path)

        return output
