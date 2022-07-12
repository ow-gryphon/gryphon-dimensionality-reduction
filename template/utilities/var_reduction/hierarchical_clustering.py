# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse
from scipy.spatial.distance import pdist
from sklearn.utils import validation

logger = logging.getLogger(__name__)


def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    From: https://github.com/scikit-learn/scikit-learn/issues/5884
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    elif X.dtype == np.object and not issparse(X):
        dtype = np.float
        for col in range(X.shape[1]):
            if not np.issubdtype(type(X[0, col]), np.number):
                dtype = np.object
                break
    else:
        dtype = np.float

    return X, Y, dtype


def check_pairwise_arrays(x: np.ndarray, y: np.ndarray, precomputed=False, dtype=None):
    """
    From: https://github.com/scikit-learn/scikit-learn/issues/5884
    :param x: First array to compare to Y
    :param y: Array to compare to X
    :param precomputed: Shape of array
    :param dtype: Data type
    :return: X, Y
    """
    x, y, dtype_float = _return_float_dtype(x, y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if y is x or y is None:
        x = y = validation.check_array(
            x, accept_sparse='csr', dtype=dtype,
            warn_on_dtype=warn_on_dtype, estimator=estimator)
    else:
        x = validation.check_array(
            x, accept_sparse='csr', dtype=dtype,
            warn_on_dtype=warn_on_dtype, estimator=estimator)
        y = validation.check_array(
            y, accept_sparse='csr', dtype=dtype,
            warn_on_dtype=warn_on_dtype, estimator=estimator)

    if precomputed:
        if x.shape[1] != y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (x.shape[0], x.shape[1], y.shape[0]))
    elif x.shape[1] != y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             x.shape[1], y.shape[1]))

    return x, y


# for mixed, ignore categorical_features
def gower_distances(x, y=None, w=None, categorical_features=None):
    """
    From https://github.com/scikit-learn/scikit-learn/issues/5884
    Computes the gower distances between X and Y
    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)

    y : array-like, shape (n_samples, n_features)

    w:  array-like, shape (n_features)
    According the Gower formula, w is an attribute weight.

    categorical_features: array-like, shape (n_features)
    Indicates with True/False wheter a column is a categorical attribute.
    This is useful when categorical atributes are represented as integer
    values.

    Returns
    -------
    similarities : ndarray, shape (n_samples, )

    Notes
    ------
    Gower is a similarity measure for categorical, boolean and numerical mixed
    data.

    """

    x, y = check_pairwise_arrays(x, y, dtype=(np.object, None)[issparse(x) or
                                                               issparse(y)])

    rows, cols = x.shape

    if categorical_features is None:
        categorical_features = []
        for col in range(cols):
            if np.issubdtype(type(x[0, col]), np.number):
                categorical_features.append(False)
            else:
                categorical_features.append(True)
    # Calculates the normalized ranges and max values of numeric values
    ranges_of_numeric = [0.0] * cols
    max_of_numeric = [0.0] * cols
    for col in range(cols):
        if not categorical_features[col]:
            max = None
            min = None
            if issparse(x):
                col_array = x.getcol(col)
                max = col_array.max() + 0.0
                min = col_array.min() + 0.0
            else:
                col_array = x[:, col].astype(np.double)
                max = np.nanmax(col_array)
                min = np.nanmin(col_array)

            if np.isnan(max):
                max = 0.0
            if np.isnan(min):
                min = 0.0
            max_of_numeric[col] = max
            ranges_of_numeric[col] = (1 - min / max) if (max != 0) else 0.0

    if w is None:
        w = [1] * cols

    yrows, ycols = y.shape

    dm = np.zeros((rows, yrows), dtype=np.double)

    for i in range(0, rows):
        j_start = i

        # for non square results
        if rows != yrows:
            j_start = 0

        for j in range(j_start, yrows):
            sum_sij = 0.0
            sum_wij = 0.0
            for col in range(cols):
                value_xi = x[i, col]
                value_xj = y[j, col]

                if not categorical_features[col]:
                    if max_of_numeric[col] != 0:
                        value_xi = value_xi / max_of_numeric[col]
                        value_xj = value_xj / max_of_numeric[col]
                    else:
                        value_xi = 0
                        value_xj = 0

                    if ranges_of_numeric[col] != 0:
                        sij = abs(value_xi - value_xj) / ranges_of_numeric[col]
                    else:
                        sij = 0
                    wij = (w[col], 0)[np.isnan(value_xi) or np.isnan(value_xj)]
                else:
                    sij = (1.0, 0.0)[value_xi == value_xj]
                    wij = (w[col], 0)[value_xi is None and value_xj is None]
                sum_sij += (wij * sij)
                sum_wij += wij

            if sum_wij != 0:
                dm[i, j] = (sum_sij / sum_wij)
                if j < rows and i < yrows:
                    dm[j, i] = dm[i, j]

    return dm


def cramers_corrected_stat(column1, column2):
    """ Calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cramérs-coefficient-matrix
    :param column1, column2: Columns of data to calculate Cramer's V
    :return: Cramer's V stat

    """
    confusion_matrix = pd.crosstab(column1, column2)
    chi2 = ss.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def run_cramer(data, na_treatment='pairwise_complete'):
    """
    Calculates Cramers V's distance matrix
    :param data: dataframe
    :param na_treatment: Either 'complete' or 'pairwise complete'
    :param result: Either 'whole' or 'condensed'
    :return: distance metric
    """

    num_var = len(data.columns)

    temp_results = pd.DataFrame(np.ones((num_var, num_var,)))
    temp_condensed = np.ones(int(num_var * (num_var-1) / 2))

    if na_treatment == "complete":
        data = data.dropna()

    counter = 0
    for item_1 in range(0, num_var - 1):
        for item_2 in range(item_1 + 1, num_var):
            var1 = data.columns.values[item_1]
            var2 = data.columns.values[item_2]

            temp_data = data[[var1, var2]]
            if na_treatment == "pairwise_complete":
                temp_data = temp_data.dropna()

            if temp_data.shape[0] == 0:
                temp_CV = np.nan
            else:
                temp_CV = cramers_corrected_stat(temp_data[var1], temp_data[var2])

            # todo error handling

            temp_results.iloc[item_1, item_2] = temp_CV
            temp_results.iloc[item_2, item_1] = temp_CV
            temp_condensed[counter] = temp_CV

            counter = counter + 1

    return temp_results, temp_condensed


def categorical_hierarchical(pd_data, linkage_method="average", metric='cramer', na_treatment='pairwise_complete'):
    """
    :param pd_data: pandas dataset with only the variables to be used for clustering
    :param linkage: One of 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'. See
    http://scipy.github.io/devdocs/generated/scipy.cluster.hierarchy.linkage.html
    Also see: https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering
    :param metric: currently only 'cramer' for adjusted cramer's v is supported
    :param na_treatment: Either 'complete' or 'pairwise complete'
    :return: clustering information, and dendrogram plot
    """

    if metric == 'cramer':
        result_matrix, result_vector = run_cramer(pd_data, na_treatment = na_treatment)
        result = linkage(1-result_vector, linkage_method)
        # Use 1 - Cramer as suggested in http://www.econ.upf.edu/~michael/stanford/maeb6.pdf
    else:
        raise ValueError("Currently only Cramer's V is available")

    return result


def get_gower(x, categorical_vars=None, weight=None):
    """
    :param x: pandas dataframe with the variables for clustering
    :param categorical_vars: list of names of categorical variables. If left as None, these will be automatically detected
    :param weight: optional weight variable (array-like, not a string)
    :return:
    """

    if categorical_vars:
        # Convert to true / false
        categoricals = [var_name in categorical_vars for var_name in x.columns.values]
    else:
        categoricals = None

    d = gower_distances(x, w=weight, categorical_features=categoricals)

    # Get a flattened upper triangular matrix
    num_var = len(x.columns)
    temp_condensed = np.ones(int(num_var * (num_var - 1) / 2))

    counter = 0
    for item_1 in range(0, num_var - 1):
        for item_2 in range(item_1 + 1, num_var):
            temp_condensed[counter] = d[item_1, item_2]
            counter = counter + 1

    return d, temp_condensed


def mixed_hierarchical(pd_data, categorical_vars=None, weight=None, linkage_method="average", metric='gower'):
    """
    :param pd_data: pandas dataset with only the variables to be used for clustering
    :param categorical_vars: list of names of categorical variables. If left as None, these will be automatically detected.
    Right now assumes that the categorical variables are passed in as strings ('cat','dog')
    if they are listed as numbers they will be treated as numeric values instead
    :param linkage: One of 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'. See
    http://scipy.github.io/devdocs/generated/scipy.cluster.hierarchy.linkage.html
    Also see: https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering
    :param metric: currently only 'gower' for gower's distance is supported
    :param weight: currently only 'gower' for gower's distance is supported
    :param linkage_method: currently only 'gower' for gower's distance is supported
    :return: clustering information, and dendrogram plot
    """

    if metric == 'gower':
        result_matrix, result_vector = get_gower(pd_data, categorical_vars, weight)
        result = linkage(result_vector, linkage_method)
    else:
        raise ValueError("Currently only gower's distance is available")

    return result


def numerical_hierarchical(pd_data, linkage_method='single', metric='euclidean', minkowski_p = None, weight = None):
    """
    :param pd_data: pandas dataset with only the variables to be used for clustering
    :param linkage_method: One of 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'. See
    http://scipy.github.io/devdocs/generated/scipy.cluster.hierarchy.linkage.html
    Also see: https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering
    :param metric: distance metric to use in the case that pd_data is a collection of observation vectors
    :return: clustering information encoded as a linkage matrix
    """
    matrix_t = pd_data.values.transpose()

    if metric == 'euclidean':
        result = linkage(matrix_t, method=linkage_method, metric="euclidean")
    elif metric in ["correlation", "cityblock", "seuclidean", "sqeuclidean", "hamming", "jaccard",
                    "chebychev", "canberra", "braycurtis", "mahalanobis", "yule", "dice", "kulsinski",
                    "rogerstanimoto", "russellrao", "sokalmichener" ,"sokalsneath"]:

        # Generate spatial distance function
        y = pdist(matrix_t, metric=metric)
        result = linkage(y, method=linkage_method)

    elif metric == "minkowski":
        y = pdist(matrix_t, metric=metric, p=float(minkowski_p))
        result = linkage(y, method=linkage_method)

    elif metric == "wminkowski":
        y = pdist(matrix_t, metric=metric, p=float(minkowski_p), w=weight)
        result = linkage(y, method=linkage_method)

    else:
        raise ValueError("Currently only euclidean distance is available")

    return result
