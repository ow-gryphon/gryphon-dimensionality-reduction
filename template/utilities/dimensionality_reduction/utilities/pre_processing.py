import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_variables(data: pd.DataFrame):
    scale = StandardScaler().fit(data)
    no_na_dataset = data.dropna()
    return pd.DataFrame(
        data=scale.transform(no_na_dataset),
        index=no_na_dataset.index,
        columns=data.columns
    )
