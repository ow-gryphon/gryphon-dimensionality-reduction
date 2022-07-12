import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing

from ..tool_utilities import column_checks
from ..tool_utilities import file_utilities

logger = logging.getLogger(__name__)


# fixed random seed=42 for clustering algorithms
def get_random_state():
    return 42


def set_run_name(run_name, timestamp):

    if (run_name is None) and (timestamp is not None):
        run_name_new = timestamp
    else:
        run_name_new = run_name

    return run_name_new


def load_data(data_dir, dataset_name, data_file_type, var_names=None, warning_info="", remove_na_inf=True,
              get_index=False):
    try:
        if data_file_type == "parquet":
            dataset_name_with_ext = dataset_name + ".parquet"
            loaded_data = file_utilities.pq_load_as_pd(data_dir, dataset_name_with_ext)
        else:
            # dataset_name_with_ext = dataset_name + "." + data_file_type
            logger.info("Non-Parquet data functionality not yet available")
            warning_info += "Non-Parquet data functionality not yet available"
            return None, None, None, warning_info
            # Only Parquet files able to be loaded
    except FileNotFoundError:
        logger.info("Error in loading file - file not found")
        warning_info += "Error in loading file - file not found"
        return None, None, None, warning_info

    # If particular variables were requested
    if var_names is not None:
        available_variables, unavailable_variables = column_checks.get_available_unavailable(
            loaded_data.columns.values,
            var_names
        )

        for var in unavailable_variables:
            print("{feat} not in data frame, and has not been loaded".format(feat=var))

        loaded_data_keep = loaded_data[available_variables]

        # check to see if there is an issue reading in unavailable variables
        if len(unavailable_variables) > 0:
            warning_info += "Some variables were unable to be read in: {}".format(unavailable_variables.tolist())
    else:
        available_variables = loaded_data.columns.values
        unavailable_variables = list()
        loaded_data_keep = loaded_data

    if remove_na_inf:
        loaded_data_keep, warning_info, keep_index = remove_missing_inf(loaded_data_keep, True, warning_info)
    else:
        keep_index = None

    if get_index:
        return loaded_data_keep, unavailable_variables, available_variables, keep_index, warning_info
    else:
        return loaded_data_keep, unavailable_variables, available_variables, warning_info


def remove_missing_inf(frame, remove_inf=True, warning_info=""):
    """
    :param frame:
    :param warning_info:
    :param remove_inf:
    :return: tuple with dataframe without na and without inf, as well string with warning about removed variables
    """
    # remove observations from loaded_data_keep
    df_no_missing = frame.dropna()

    # Check for missing values
    num_missing = frame.shape[0] - df_no_missing.shape[0]

    if num_missing > 0:
        # Add warning_info that rows were removed from loaded_data_keep, by taking the max of the missing values
        warning_info += "Removed {} observations from the analysis because of missing values".format(num_missing)

    if remove_inf:
        numerics = df_no_missing.select_dtypes(include=[np.number])
        numerics = numerics.replace([np.inf, -np.inf], np.nan).dropna()

        # Check for missing values
        num_inf = df_no_missing.shape[0] - numerics.shape[0]

        if num_inf > 0:
            warning_info += "Removed {} observations from the analysis because of infinite values".format(num_inf)
            inf_index = numerics.index.values
            df_no_missing = df_no_missing.iloc[inf_index, :]

    keep_index = df_no_missing.index.values
    df_no_missing = df_no_missing.reset_index(drop=True)

    return df_no_missing, warning_info, keep_index


def standardize(frame, variables=None):

    loaded_data_pre_transform = frame
    # TODO - confirm data is numerical

    if variables is None:
        scalar = preprocessing.StandardScaler().fit(frame)
        standardized_data = pd.DataFrame(scalar.transform(frame), columns=frame.columns)
        frame = standardized_data

    else:
        # assert the vars are in frame
        sub_frame = frame[variables]
        scalar = preprocessing.StandardScaler().fit(sub_frame)
        standardized_data = pd.DataFrame(scalar.transform(sub_frame), columns=sub_frame.columns)
        frame[variables] = standardized_data

    return frame, loaded_data_pre_transform
