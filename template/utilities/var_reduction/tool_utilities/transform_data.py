import pandas as pd
import numpy as np
from tool_utilities.file_utilities import *
from tool_utilities.math_utilities import *
from tool_utilities.column_checks import *
from tool_utilities.function_init import *
from tool_utilities.scan_data import *
from matplotlib import pyplot as plt


def convert_to_dummies(frame, variables, drop_first = False, dummy_na = True, prefix_sep = "_"):
    '''
    Apply get_dummies function from pandas to encode categoricals
    :param frame: pandas dataframe
    :param variables: list of categorical variables to encode
    :param drop_first: whether to drop the first level
    :param dummy_na: whether to create a dummy for NA
    :param prefix_sep: separator between variable name and value
    :return: pandas dataframe with additional columns
    '''

    new_columns = pd.get_dummies(frame[variables], columns=variables, drop_first = drop_first, dummy_na = dummy_na,
                               prefix_sep = prefix_sep)
    proposed_names = new_columns.columns.values

    # Check column names
    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns = dict(zip(proposed_names, new_names)))

    # Concatenate with original data
    new_frame = pd.concat([frame,new_columns], axis = 'columns')

    return new_frame


def get_avg_DV(frame, variables, DV, DV_trans):
    '''
    Generate numerical transformation of the categorical variable based on the average dependent variable value
    :param frame: pandas dataframe
    :param variables: list of categorical variables to encode
    :param DV: Dependent variable value
    :param DV_trans: Transformation to be applied, either "No transformation", "Log", "LogP1", "Logit", "Robust logit"
    :return: pandas dataframe with new columns
    '''

    for var in variables:
        new_column = frame[[DV,var]].copy()
        new_column.loc[new_column[var].isnull(), var] = "__Empty__"
        new_data = new_column.groupby(var).transform(lambda x: x.mean())
        new_data.columns = [var]

        if DV_trans == "No transformation":
            proposed_name = "{}_avgDV".format(var)
        elif DV_trans == "Log":
            new_data[var] = np.log(new_data[var])
            proposed_name = "{}_avgDVlog"
        elif DV_trans == "LogP1":
            new_data[var] = np.log(1.0 + new_data[var])
            proposed_name = "{}_avgDVlogP1"
        elif DV_trans == "Logit":
            new_data[var] = np.log(new_data[var] / (1-new_data[var]))
            proposed_name = "{}_avgDVlogit"
        elif DV_trans == "Robust Logit":
            counts = new_column.groupby(var).transform(lambda x: x.size())
            new_data.loc[new_data[var] == 0,:] = 0.5 / counts[var][new_data == 0]
            new_data.loc[new_data[var] == 1,:] = 1 - 0.5 / counts[var][new_data == 1]
            new_data[var] = np.log(new_data[var] / (1 - new_data[var]))
            proposed_name = "{}_avgDVroblogit"
        else:
            raise ValueError("No transformation possible for variable {}".format(var))

        new_name = generate_column_name(proposed_name, frame)
        frame[new_name] = new_data[var]

    return frame


def standardize(frame, variables, type, suffix = None):
    '''
    Standardize numerical variables in dataset
    :param frame: pandas dataframe
    :param variables: list of numerical variables to standardize
    :param type: type of standardization: 'standardize', 'scale', 'demean', '[0,1]','[-1,1]
    :param suffix: suffix for new variables. If None, will use '_std','_scale', '_center', '_unit', '_unit2'
    :return: pandas dataframe with new columns
    '''

    new_columns = frame[variables]
    orig_names = new_columns.columns.values

    if type == "standardize":
        new_columns = (new_columns - new_columns.mean()) / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_std".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "scale":
        new_columns = new_columns / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_scale".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "demean":
        new_columns = new_columns - new_columns.mean()
        if suffix is None:
            proposed_names = ["{}_center".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "[0,1]":
        new_columns = (new_columns - new_columns.min()) / (new_columns.max() - new_columns.min())
        if suffix is None:
            proposed_names = ["{}_unit".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "[-1,1]":
        new_columns = 2*(new_columns - new_columns.min()) / (new_columns.max() - new_columns.min()) - 1
        if suffix is None:
            proposed_names = ["{}_unit2".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]

    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns=dict(zip(orig_names, new_names)))
    new_frame = pd.concat([frame, new_columns], axis='columns')

    return new_frame


def convert_num_to_cat(frame, variables, suffix = "_cat"):
    '''
    Convert numerical variables to categorical variables
    :param frame: pandas dataframe
    :param variables: variables to convert to categorical
    :param suffix: suffix for variable
    :return: pandas data frame with categorical variables
    '''

    #new_columns = frame[variables].apply(lambda x: x.astype('category'))
    new_columns = frame[variables].apply(lambda x: x.astype('str'))
    orig_names = new_columns.columns.values
    proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns=dict(zip(orig_names, new_names)))
    new_frame = pd.concat([frame, new_columns], axis='columns')

    return new_frame


def convert_cat_to_num(frame, variables, suffix="_num"):
    '''
    Convert categorical variables to numerical variables
    :param frame: pandas dataframe
    :param variables: variables to convert to categorical
    :param suffix: suffix for variable
    :return: pandas data frame with categorical variables
    '''

    new_columns = frame[variables].apply(lambda x: x.astype('float'))
    orig_names = new_columns.columns.values
    proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns=dict(zip(orig_names, new_names)))
    new_frame = pd.concat([frame, new_columns], axis='columns')

    return new_frame


def data_transform(work_dir, data_dir, dataset_name,variables, transform, option = None, timestamp = None,
                   run_name = None, data_file_type = "parquet", selected_DV = None, selected_DV_transformation = None,
                         status_file = None, warning_info = "", only_return = False):
    '''
    Perform transformations of variables in the dataset (see the transformation argument)
    :param work_dir: Working dir
    :param data_dir: Data dir
    :param dataset_name: Name of dataset
    :param variables: Variables for transformation
    :param transform: type of transformation:
        encode_cat: encode categorical variable into numerical
        standardize: standardize variable
        convert_num_to_cat: convert numerical type to categorical
        convert_cat_to_num: convert categorical type to numerical
    :param option:
        for encode_cat: "Hot Encode", "Hot Encode - 1", "Avg DV"
        for standardize: "standardize", "scale", "demean", "[0,1]", "[-1,1]"
        for others: None
    :param timestamp: Timestamp of run
    :param run_name: Name of run
    :param data_file_type: File type where dataset is located
    :param selected_DV: If Avg DV is selected for encoding of categorical, this is the name of the categorical value
    :param selected_DV_transformation: If Avg DV is selected for encoding of categorical, this transformation is used
        either: "No transformation", "Log", "LogP1", "Logit", "Robust logit"
    :param status_file: File to track error logs
    :param warning_info: Warning info to pass
    :param only_return: boolean whether to return results rather than to save it into json
    :return: warning_info
    '''

    set_status_file(status_file, "Load in data and settings")
    run_name = set_run_name(run_name, timestamp)

    loaded_data_keep, unavailable_variables, available_variables, warning_info = load_data(data_dir, dataset_name,
                                                                                           data_file_type,
                                                                                           warning_info=warning_info,
                                                                                           remove_na_inf = False)

    # Check if any variables for transformation are missing
    use_variables = [var for var in variables if var in available_variables]

    if len(use_variables) < len(variables):
        not_available = list(set(variables) - set(use_variables))
        warning_info = warning_info + "Following variables not available {}/n".format(', '.join(not_available))

    set_status_file(status_file, "Running transformations")
    # Perform transformations
    if transform == "encode_cat":

        if option == "Hot Encode":
            transformed_data = convert_to_dummies(loaded_data_keep, use_variables, drop_first=False)

        elif option == "Hot Encode - 1":
            transformed_data = convert_to_dummies(loaded_data_keep, use_variables, drop_first=True)

        elif option == "Avg DV":
            transformed_data = get_avg_DV(loaded_data_keep, use_variables,
                                                      DV = selected_DV, DV_trans = selected_DV_transformation)

    elif transform == "standardize":
        transformed_data = standardize(loaded_data_keep, use_variables, option)

    elif transform == "convert_num_to_cat":
        transformed_data = convert_num_to_cat(loaded_data_keep, use_variables)

    elif transform == "convert_cat_to_num":
        transformed_data = convert_cat_to_num(loaded_data_keep, use_variables)

    # Save results, overwriting parquet
    set_status_file(status_file, "Saving new dataset")
    try:
        save_path = data_dir
        save_name = dataset_name
        pd_save_as_pq(transformed_data, save_path, save_name)
        logger.info("Vars saved")
    except:
        warning_info = warning_info + "Unable to save dataset\n"
        if only_return:
            return None, warning_info
        else:
            return warning_info

    # Scan dataset
    set_status_file(status_file, "Scanning new dataset")
    scan_datasets(work_dir, data_dir, timestamp=timestamp, datasets=[dataset_name],
                  status_file=status_file, warning_info=warning_info)

    if only_return:
        return transformed_data, warning_info
    else:
        return warning_info

