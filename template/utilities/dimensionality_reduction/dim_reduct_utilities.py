import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
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



def export_plot(figure: plt.figure, prefix="figure", output_path: Path = Path.cwd(), **kwargs):
    timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())
    file_name = f"{prefix}_{timestamp}.png"

    if not output_path.is_dir():
        os.makedirs(output_path, exist_ok=True)

    figure.savefig(
        output_path / file_name,
        **kwargs
    )



def pd_save_as_xlsx(frame, save_path, save_name, show_index=True, sheet_name=None, writer=None):
    """
    Saves pd as xlsx
    :param frame: pandas dataframe
    :param save_path: path to directory where the data to be loaded is currently saved
    :param save_name: name of the dataset file to be loaded, which should have the file extension .parquet
    :param show_index: Boolean to include index in excel
    :param sheet_name: name of sheet
    :param writer: sets existing writer
    :return: writer used for xlsx
    """

    if sheet_name is None:
        sheet_name = save_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if writer is None:
        full_save_ref = os.path.join(save_path, save_name)
        writer = pd.ExcelWriter("{}.xlsx".format(full_save_ref))

    try:
        frame.to_excel(writer, sheet_name=sheet_name, index=show_index)
    except Exception:
        # truncate run_name if too long
        frame.to_excel(writer, sheet_name=sheet_name[0:15], index=show_index)
    return writer