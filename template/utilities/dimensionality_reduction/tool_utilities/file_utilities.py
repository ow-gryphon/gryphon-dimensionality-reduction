# Utilities for saving and reading parquet files, and converting between pandas and parquet

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def export_plot(figure: plt.figure, prefix="figure", output_path: Path = Path.cwd(), **kwargs):
    timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())
    file_name = f"{prefix}_{timestamp}.png"

    if not output_path.is_dir():
        os.makedirs(output_path, exist_ok=True)

    figure.savefig(
        output_path / file_name,
        **kwargs
    )


def pd_save_as_pq(pd_data, save_path, save_name):
    """
    Converts pandas data frame into a pyarrow table and then saves it as a .parquet file

    :param pd_data: a single pandas DataFrame to be saved
    :param save_path: path to directory where data should be saved
    :param save_name: name of the dataset to be saved
    :return: none, but saves parquet file to location specified
    """
    table_to_save = pa.Table.from_pandas(pd_data)
    pq.write_table(table_to_save, os.path.join(save_path, '{}.parquet'.format(save_name)))

    return None


def pq_load_as_pd(save_path, save_name):
    """
    Loads in a .parquet file and converts it to a pandas data frame into a pyarrow table and then saves it as
    a .parquet file
    :param save_path: path to directory where the data to be loaded is currently saved
    :param save_name: name of the dataset file to be loaded, which should have the file extension .parquet
    :return: Pandas dataframe
    """

    # with open(os.path.join(save_path, save_name)) as f:
    loaded_table = pq.read_table(os.path.join(save_path, save_name))

    return loaded_table.to_pandas()


def csv_load_as_pd(save_path, save_name, delete = False):
    """
    Loads csv file as pandas dataframe
    :param save_path: name of dir where data is loaded
    :param save_name: name of file to be loaded
    :param delete: delete csv afterwards?
    :return: pandas dataframe
    """

    try:
        loaded_table = pd.read_csv(os.path.join(save_path, save_name),low_memory=False)
    except UnicodeDecodeError:  # add UnicodeDecodeError exception to handle some characters (é)
        loaded_table = pd.read_csv(os.path.join(save_path, save_name), encoding='latin-1',low_memory=False)

    if delete:
        try:
            os.unlink(os.path.join(save_path, save_name))
        except:
            pass

    return loaded_table


def save_as_png(plot_obj, save_path, save_name):
    """
    Saves plot object as .png file
    :param save_path: name of dir where data is loaded
    :param save_name: name of file to be loaded
    :return: none, but saves .png file
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_obj.savefig("{}{}".format(save_path, save_name), bbox_inches='tight')
    return


def pd_save_as_csv(frame, save_path, save_name):
    """
    Saves pd as csv
    :param frame: pandas dataframe
    :param save_path: path to directory where the data to be loaded is currently saved
    :param save_name: name of the dataset file to be loaded, which should have the file extension .parquet
    :return: None, but saves CSV
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    frame.to_csv(os.path.join(save_path, save_name))
    return


def set_status_file(status_file,message):
    """
    Sets the status_file to a message
    :param status_file: Status file to set
    :param message: Message to display
    :return: Status file, saves message to status file
    """
    if status_file is not None:
        status_file = open(status_file.name, 'w')
        status_file.write(message)
        status_file.close()
    return status_file


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


def export_data(work_dir, data_dir, dataset, timestamp="", data_file_type="parquet",
                status_file=None, warning_info="", only_return=False):
    """
    Export data file as csv
    :param work_dir: Working dir
    :param data_dir: Data dir
    :param dataset: Name of dataset
    :param timestamp: Timestamp of run
    :param data_file_type: File type where dataset is located
    :param status_file: File to track error logs
    :param warning_info: Warning info to pass
    :return: String with location of the Excel file
    """

    set_status_file(status_file, "Opening dataset")
    if data_file_type == "parquet":
        loaded_data = pq_load_as_pd(data_dir, "{}.parquet".format(dataset))

    else:
        warning_info = warning_info + "Dataset is not a parquet dataset\n"

        if only_return:
            return None, warning_info
        else:
            return warning_info

    set_status_file(status_file, "Save dataset as csv")
    save_path = work_dir + "/results"
    save_name = "{}_{}.csv".format(dataset, timestamp)
    loaded_data.to_csv(os.path.join(save_path, save_name))

    filepath = "{}/{}_{}.csv".format(save_path, dataset, timestamp)
    os.startfile(filepath)

    if only_return:
        return filepath, warning_info
    else:
        return warning_info
