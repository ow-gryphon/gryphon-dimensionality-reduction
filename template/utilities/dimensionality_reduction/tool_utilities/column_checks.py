def generate_column_name(proposed_name, frame):
    """
    Generate a variable name that is not already in the dataset

    :param proposed_name: string representing the proposed name of the variable
    :param frame: pandas dataframe for which the new variable name should be used
    :return: string with a non-duplicated and appropriate name for the dataframe
    """

    orig_name = proposed_name.strip().replace(' ', '_').replace('(', '').replace(')', '')
    proposed_name = orig_name
    num = 1
    while proposed_name in frame.columns.values:
        num = num + 1
        proposed_name = "{}_{}".format(orig_name, num)

    return proposed_name


def get_available_unavailable(existing_vars, requested_vars):
    """
    :param existing_vars: List of variable names from the data frame
    :param requested_vars: List of variables to check for existence
    :return: tuple with available variables and unavailable variables (with original ordering of the variables)
    """

    unavailable_variables = list(set(requested_vars) - set(existing_vars))
    available_variables = list(set(requested_vars) - set(unavailable_variables))

    unavailable_variables = [var for var in requested_vars if var in unavailable_variables]
    available_variables = [var for var in requested_vars if var in available_variables]

    return available_variables, unavailable_variables
