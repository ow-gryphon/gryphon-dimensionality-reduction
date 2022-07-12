import glob
import json
import os

import numpy as np
from pandas.api.types import is_numeric_dtype

# Tool specific files
from ..tool_utilities import file_utilities
from ..tool_utilities import math_utilities


def scan_datasets(work_dir, data_dir, timestamp=None, datasets=None, status_file=None, warning_info=""):
    """
    Obtain metadata from all (or specified) parquet files available in the data directory, and save the metadata as a
    json file in the working directory for the tool to consume.

    :param work_dir: working directory path
    :param data_dir: data directory path
    :param timestamp: date and time stamp for the json file to be saved. When using the tool, the Node.js server will
    generate this timestamp and will expect the output json file to be named accordingly
    :param datasets: this is an optional list of dataset names to be scanned. If None it will scan all found datasets.
    :param status_file:
    :param warning_info:
    :return: None, but it saves a .json with summary information
    """

    status_file = open(status_file.name, 'w')
    status_file.write('Identifying and scanning datasets')
    status_file.close()

    # Find parquet datasets in data_dir folder
    os.chdir(data_dir)
    parquet_files = glob.glob("*.parquet")

    if datasets is not None:
        if type(datasets) is list:
            raise TypeError("The function argument 'datasets' is not a LIST of dataset names")
        parquet_files = list(
            set(parquet_files).intersection(["{}.parquet".format(data_name) for data_name in datasets])
        )
        if len(parquet_files) == 0:
            raise TypeError("No dataset with the requested name")

    summary_output = list()

    for parquet_file in parquet_files:
        temp_data = file_utilities.pq_load_as_pd(data_dir, parquet_file)
        temp_data_info = scan_file(temp_data, os.path.splitext(parquet_file)[0])
        summary_output.append(temp_data_info)

    csv_files = glob.glob("*.csv")

    for csv_file in csv_files:
        temp_data = file_utilities.csv_load_as_pd(data_dir, csv_file, delete=True)
        temp_data_info = scan_file(temp_data, os.path.splitext(csv_file)[0])
        summary_output.append(temp_data_info)

        # saving all csv files as PQs
        file_utilities.pd_save_as_pq(temp_data, data_dir, os.path.splitext(csv_file)[0])

    # Save output
    output_file_path = os.path.join(work_dir, "python_json", "scan_data_{}.json".format(timestamp))
    with open(output_file_path, 'w') as outfile:
        json.dump(summary_output, outfile)

    return warning_info


def scan_file(frame, dataset_name):
    """

    :param frame: data frame to scan
    :param dataset_name: name of dataset
    :return: Dictionary of key dataset info
    """

    temp_data_info = dict()

    # Name of dataset
    temp_data_info["Data"] = dataset_name

    # Number of variables
    temp_data_info["Vars"] = frame.shape[1]

    # Number of observations
    temp_data_info["Rows"] = frame.shape[0]

    # Other information about the datasets, such as description and composition are not read in here

    # Scan the variables within the dataset
    var_info, var_scan_type = scan_variables(frame)
    temp_data_info["Variables"] = var_info
    temp_data_info["Variables_info"] = var_scan_type

    return temp_data_info


def scan_variables(df_data, variables=None):
    """
    Obtain metadata for the variables in a given dataset, with the option to restrict the scanning to certain
    variables only

    :param df_data: pandas dataframe with variables to be scanned
    :param variables: list of variables to be scanned. If None, will scan all variables
    :return: a tuple containing (i) list of dictionaries of meta data and (ii) information about the type of scan
     performed
    """
    if variables is None:
        variables = df_data.columns.values.tolist()
        scan_type = "All"
    else:
        variables = set(variables).intersection(df_data.columns.values.tolist())
        scan_type = "Specified"

    # Perform list comprehension to get the information from the variables
    variable_info = [scan_single_variable(df_data, variable) for variable in variables]

    return variable_info, scan_type


def scan_single_variable(df_data, variable):
    """
    Helper code for the 'scan_variables' function, which scans a single variable.

    :param df_data: pandas dataframe with variables to be scanned
    :param variable: name of variable to be scanned
    :return: a dictionary with meta data for the variable
    """

    if variable not in df_data.columns.values.tolist():
        raise ValueError("Variable {} is not in dataset".format(variable))

    def convert_inf(x):
        if x == float("inf"):
            return "Inf"
        elif x == float("-inf"):
            return "-Inf"
        else:
            return x

    if is_numeric_dtype(df_data[variable]):
        uniques = df_data[variable].unique()
        if all(np.isnan(uniques)):
            unique10 = ["All NaN"]
            the_range = "All NaN"
        else:
            min_val = math_utilities.round_sig(np.asscalar(np.nanmin(uniques)), 3)
            min_val = convert_inf(min_val)
            max_val = math_utilities.round_sig(np.asscalar(np.nanmax(uniques)), 3)
            max_val = convert_inf(max_val)
            unique10 = [convert_inf(math_utilities.round_sig(x, 3)) for x in uniques[0:10].tolist()]
            the_range = "{} to {}".format(min_val, max_val)

        return {
            "Name": variable,
            "Class": df_data[variable].dtype.name,
            "Missing": np.asscalar(df_data[variable].isnull().sum()),
            "NumUnique": uniques.shape[0],
            "Unique10": unique10,
            "Range": the_range
        }
    else:
        uniques = df_data[variable].unique()
        unique10 = uniques[0:10].tolist()
        non_na_unique = list(filter(None, uniques))
        non_na_unique = [str(val) for val in non_na_unique]

        return {
            "Name": variable,
            "Class": df_data[variable].dtype.name,
            "Missing": np.asscalar(df_data[variable].isnull().sum()),
            "NumUnique": uniques.shape[0],
            "Unique10": ['' if s is None else str(s) for s in unique10],
            "Range": "{} to {}".format(str(min(non_na_unique)), str(max(non_na_unique)))
        }
