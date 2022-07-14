# -*- coding: utf-8 -*-
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .hierarchical_clustering import numerical_hierarchical, categorical_hierarchical, mixed_hierarchical
from .utilities import file_utilities
from .utilities import pre_processing

logger = logging.getLogger(__name__)

possible_metrics = [
    "minkowski", "wminkowski", "correlation", "cityblock", "seuclidean", "sqeuclidean", "hamming", "jaccard",
    "chebychev", "canberra", "braycurtis", "mahalanobis", "yule", "dice", "kulsinski", "rogerstanimoto",
    "russellrao", "sokalmichener", "sokalsneath"
]


class AgglomerativeFeatureClustering:
    @staticmethod
    def agg_cluster(
            data, variables,
            minkowski_p=None, linkage=None, weight_var=None,
            distance_metric=None, categorical_vars=None,
            standardize_vars=False,
            generate_charts=True,
            save_results_to_excel=True,
            output_path=Path.cwd() / "outputs"
    ):

        """
        Creates a dendrogram using agglomerative clustering.  Tool specific function to call the relevant python code.

        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        #Perform-the-Hierarchical-Clustering

        :param data: Dataset
        :param variables: Variables to be included in dendrogram
        :param minkowski_p: Name of run
        :param linkage: Name of run
        :param weight_var: Name of run
        :param standardize_vars: boolean - if selected, variables should be standardized
        :param distance_metric: informs which clustering algorithm to run and passes the appropriate parameters
        :param categorical_vars: list of categorical variables if supposed to be categorical instead of numeric
        :param save_results_to_excel: boolean whether to save Excel
        :param generate_charts: boolean whether to save png file
        :param output_path: boolean whether to save png file

        :return: warning_info
        """
        # ----------Set-up--------------
        timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())

        if len(variables) == 1:
            raise ValueError('ERROR: More than one variable is required for dendrogram.')

        weight_values = None
        if weight_var is not None:
            weight_values = data[weight_var]
            variables = variables + [weight_var]

        dataset = data[variables]

        if weight_var is not None and weight_var not in dataset.columns.values:
            raise ValueError('ERROR: The weight variable was not available.')

        dataset = dataset[[
            var
            for var in dataset.columns.values
            if var != weight_var
        ]]

        # Only standardize for pure numeric
        if standardize_vars:
            dataset = pre_processing.standardize_variables(dataset)

        # ------------ Call particular function based on the distance metric you want- -------------------
        other_information = None  # The purpose of this is to capture other meta information about the execution

        if distance_metric == 'euclidean':
            print('Running basic numerical hierarchical clustering')
            if linkage is None:
                linkage = 'single'
            z = numerical_hierarchical(dataset, metric=distance_metric, linkage_method=linkage)
            # can define more functions here
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        elif distance_metric in possible_metrics:
            print('Running advanced hierarchical clustering')

            if weight_var is None:
                weight_values = None
                if distance_metric == "wminkowski":
                    raise AttributeError("ERROR: No weight provided.")

            z = numerical_hierarchical(
                dataset,
                metric=distance_metric,
                minkowski_p=minkowski_p,
                linkage_method=linkage,
                weight=weight_values.values
            )

            if distance_metric in ["minkowski", "wminkowski"]:
                other_information = f"Minkowski order {minkowski_p}"

        elif distance_metric == 'cramer':
            print('Running categorical hierarchical clustering')
            linkage = "average"
            z = categorical_hierarchical(dataset, metric=distance_metric, linkage_method=linkage)

        elif distance_metric == 'gower':
            print('Running mixed hierarchical clustering')
            linkage = "average"
            z = mixed_hierarchical(dataset, metric=distance_metric, categorical_vars=categorical_vars,
                                   linkage_method=linkage)
            # categorical_vars just in case categorical variable is passed as a numeric.

        else:
            raise AttributeError('ERROR: Invalid distance metric used.')

        if generate_charts:
            # static size that looks OK plt.figure(figsize=(25, 10))
            var_num = len(dataset.columns)
            figure = plt.figure(figsize=(var_num * 1.4, var_num * 0.6))

            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Variable')
            plt.ylabel('Distance')

            dendrogram(
                z,
                leaf_rotation=90.,  # rotates the x axis labels - automatic rotation looked bad
                leaf_font_size=8,  # font size for the x axis labels - automatic rotation looked bad
                labels=np.array(list(map(lambda x: x[:20], dataset))),
                # truncate variable name to 20 characters max
                color_threshold=0  # turn off coloring of tree
            )

            file_utilities.export_plot(figure, output_path=output_path, prefix="dendrogram")

        # saves results to xlsx
        if save_results_to_excel:
            save_name = f"linkage_matrix_{timestamp}"
            file_utilities.pd_save_as_xlsx(pd.DataFrame(z), output_path, save_name)

        print('End Agglo Feature Clustering')

        output = dict()
        output['distance'] = z
        output['dendrogram'] = plt
        output["model_type"] = "Reduction Existing"
        output["distance_metric"] = distance_metric
        output["weight_var"] = weight_var
        output["other info"] = other_information

        return output
