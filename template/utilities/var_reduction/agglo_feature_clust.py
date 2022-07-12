# -*- coding: utf-8 -*-
import json
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .hierarchical_clustering import numerical_hierarchical, categorical_hierarchical, mixed_hierarchical
from .tool_utilities import file_utilities
from .tool_utilities import function_init

logger = logging.getLogger(__name__)

possible_metrics = [
    "minkowski", "wminkowski", "correlation", "cityblock", "seuclidean", "sqeuclidean", "hamming", "jaccard",
    "chebychev", "canberra", "braycurtis", "mahalanobis", "yule", "dice", "kulsinski", "rogerstanimoto",
    "russellrao", "sokalmichener", "sokalsneath"
]


def agg_cluster(work_dir, data_dir, dataset_name, var_names, timestamp=None, run_name=None,
                minkowski_p=None, linkage=None, weight_var=None,
                data_file_type="parquet", status_file=None, warning_info="", standardize_vars=False,
                distance_metric="None", categorical_vars=None, save_xlsx=False, save_png=True,
                only_return=False):

    """
    Creates a dendrogram using agglomerative clustering.  Tool specific function to call the relevant python code.

    #https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Perform-the-Hierarchical-Clustering

    :param work_dir: Working dir
    :param data_dir: Data dir
    :param dataset_name: Name of dataset
    :param var_names: Variables to be included in dendrogram
    :param timestamp: Timestamp of run
    :param run_name: Name of run
    :param minkowski_p: Name of run
    :param linkage: Name of run
    :param weight_var: Name of run
    :param data_file_type: File type where dataset is located
    :param status_file: File to track error logs
    :param warning_info: Warning info to pass
    :param standardize_vars: boolean - if selected, variables should be standardized
    :param distance_metric: informs which clustering algorithm to run and passes the appropriate parameters
    :param categorical_vars: list of categorical variables if supposed to be categorical instead of numeric
    :param save_xlsx: boolean whether to save Excel
    :param save_png: boolean whether to save png file
    :param only_return: boolean whether to return results rather than to save it into json
    :return: warning_info
    """
    # ----------Set-up--------------

    run_name = function_init.set_run_name(run_name, timestamp)
    weight_values = None

    if len(var_names) == 1:
        warning_info += 'ERROR: Requires more than one variable for dendrogram'
        return warning_info

    if weight_var is None:
        loaded_data_keep, unavailable_variables, available_variables, warning_info = function_init.load_data(
            data_dir, dataset_name,
            data_file_type,
            var_names=var_names,
            warning_info=warning_info
        )
    else:
        loaded_data_keep, unavailable_variables, available_variables, warning_info = function_init.load_data(
            data_dir, dataset_name,
            data_file_type,
            var_names=var_names + [weight_var],
            warning_info=warning_info
        )

        if weight_var not in loaded_data_keep.columns.values:
            warning_info += "The weight variable was not available\n"
            return warning_info

        weight_values = loaded_data_keep[weight_var]
        loaded_data_keep = loaded_data_keep[[var for var in loaded_data_keep.columns.values if var != weight_var]]

    if loaded_data_keep is None:
        warning_info += "No data was able to be loaded \n"
        return warning_info

    # Only standardize for pure numeric
    if standardize_vars:
        try:
            loaded_data_keep, loaded_data_pre_transform = function_init.standardize(loaded_data_keep)
        except ValueError:
            logger.info('Categorical variables could not be standardized')

    # ------------Call particular function based on the distance metric you want--------------------
    other_information = None  # The purpose of this is to capture other meta information about the execution

    if distance_metric == 'euclidean':
        logger.info('Running basic numerical hierarchical clustering')
        file_utilities.set_status_file(status_file, 'Running numerical hierarchical clustering')
        if linkage is None:
            linkage = 'single'
        z = numerical_hierarchical(loaded_data_keep, metric=distance_metric, linkage_method=linkage)
        # can define more functions here
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    elif distance_metric in possible_metrics:
        logger.info('Running advanced hierarchical clustering')
        file_utilities.set_status_file(status_file, 'Running numerical hierarchical clustering')

        if weight_var is None:
            weight_values = None
            if distance_metric == "wminkowski":
                warning_info += "No weight provided \n"
                return warning_info

        z = numerical_hierarchical(
            loaded_data_keep,
            metric=distance_metric,
            minkowski_p=minkowski_p,
            linkage_method=linkage,
            weight=weight_values.values
        )

        if distance_metric in ["minkowski", "wminkowski"]:
            other_information = f"Minkowski order {minkowski_p}"

    elif distance_metric == 'cramer':
        logger.info('Running categorical hierarchical clustering')
        file_utilities.set_status_file(status_file, 'Running categorical hierarchical clustering')
        linkage = "average"
        z = categorical_hierarchical(loaded_data_keep, metric=distance_metric, linkage_method=linkage)

    elif distance_metric == 'gower':
        logger.info('Running mixed hierarchical clustering')
        file_utilities.set_status_file(status_file, 'Running mixed hierarchical clustering')
        linkage = "average"
        z = mixed_hierarchical(loaded_data_keep, metric=distance_metric, categorical_vars=categorical_vars,
                               linkage_method=linkage)
        # categorical_vars just in case categorical variable is passed as a numeric.
    else:
        logger.info('Invalid distance metric used')
        warning_info += 'Invalid distance metric used\n'
        return warning_info

    # static size that looks OK plt.figure(figsize=(25, 10))
    plt.figure(figsize=(len(loaded_data_keep.columns) * 1.4, len(loaded_data_keep.columns) * 0.6))

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Variable')
    plt.ylabel('Distance')
    dendrogram(
        z,
        leaf_rotation=90.,  # rotates the x axis labels - automatic rotation looked bad
        leaf_font_size=8,  # font size for the x axis labels - automatic rotation looked bad
        labels=np.array(list(map(lambda x: x[:20], loaded_data_keep))),  # truncate variable name to 20 characters max
        color_threshold=0  # turn off coloring of tree
    )

    # plt.show()
    png_path = f"{work_dir}/results/{run_name}_{timestamp}/png/"
    png_name = f"{run_name}_{timestamp}.png"
    # save image - defaults to True
    if save_png:
        file_utilities.save_as_png(plt, png_path, png_name)

    # saves results to xlsx
    if save_xlsx:
        save_path = f"{work_dir}/results/{run_name}_{timestamp}/xlsx/"
        save_name = f"{run_name}_linkage_matrix"
        file_utilities.pd_save_as_xlsx(pd.DataFrame(z), save_path, save_name)

    logger.info('End Agglo Feature Clustering')
    if only_return:
        temp_results = dict()
        temp_results['distance'] = z
        temp_results['dendrogram'] = plt
        warnings = None
        return temp_results, warnings

    else:
        # prepare .JSON file
        temp_data_info = dict()
        temp_data_info["dataset"] = dataset_name
        temp_data_info["model_type"] = "Reduction Existing"
        temp_data_info["distance_metric"] = distance_metric
        if weight_var is not None:
            temp_data_info["weight_var"] = weight_var

        temp_data_info["linkage"] = linkage

        if other_information is not None:
            temp_data_info["other info"] = other_information

        temp_data_info["timestamp"] = timestamp
        temp_data_info["vars_chosen"] = var_names
        temp_data_info["run_name"] = run_name
        temp_data_info["linktoPNG"] = png_path + png_name

        output_file_path = os.path.join(work_dir, "python_json", f'AggloFeatureClus_{timestamp}.json')

        with open(output_file_path, 'w') as outfile:
            json.dump(temp_data_info, outfile)

        # close out plt to save memory
        # plt.close('all')

        return warning_info

    # plt.show()
    # a = fcluster(Z, criterion="maxclust")
