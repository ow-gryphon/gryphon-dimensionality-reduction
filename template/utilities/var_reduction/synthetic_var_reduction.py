# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .tool_utilities.pre_processing import standardize_variables

logger = logging.getLogger(__name__)


def pca_var_reduction(
        data, variables, n_components,
        standardize_vars=False,
        generate_charts=False,
        save_results_to_excel=False
):

    """
    Reduce numerical variables using PCA

    :param data: DataFrame
    :param variables: variable names to be reduced
    :param n_components: maximum number of components tested
    :param generate_charts: Whether to generate charts or not
    :param save_results_to_excel: Whether to save results to excel or not
    :param standardize_vars: Boolean to standardize variables

    :return: dictionary, warning_info
    """

    dataset = data[variables]

    if standardize_vars:
        dataset = standardize_variables(dataset)

    # ------------Begin PCA-----------------------------------------

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

    temp_data_info = dict()
    temp_data_info["data"] = frame_with_pc
    temp_data_info["model"] = pca
    temp_data_info['explained_variance'] = explained_var_cs_df
    temp_data_info['components'] = res
    temp_data_info["model_type"] = "Reduction Synthetic"
    temp_data_info["variance explained"] = total_var_explained
    temp_data_info["plot data"] = np.array(explained_var_cs_df.values)
    temp_data_info["plot_type"] = "PCA"

    if save_results_to_excel:
        print("Saving PCA results")
        pass

    if generate_charts:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title("Explained variance ratio (cumulative)")
        ax.plot(
            temp_data_info["plot data"][:, 1],
            temp_data_info["plot data"][:, 0],
            marker='.',
            markersize=10
        )
        ax.set_xlabel("Component num")
        ax.set_ylabel("Ratio")
        ax.set_ylim([0, 1])
        ax.set_yticks(np.linspace(0, 1, 11))
        plt.grid()

    return temp_data_info
