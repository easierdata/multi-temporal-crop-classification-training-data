import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List

import sys

module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


def create_empty_df() -> pd.DataFrame:
    """
    Creates an empty DataFrame with predefined columns.

    The DataFrame will have a "chip_id" column followed by columns named
    after the keys in the CLASS_NAMES dictionary.

    Returns:
        pd.DataFrame: An empty DataFrame with the specified columns.
    """
    df_columns = ["chip_id"] + list(range(CROP_CLASS_MAPPING.keys()))
    return pd.DataFrame(columns=df_columns)


def filter_files_by_chip_ids(chip_ids: List[str], files: List[Path]) -> List[Path]:
    """
    Filters a list of file paths by checking if any of the specified chip IDs are present in the file names.

    Args:
        chip_ids (List[str]): A list of chip IDs to filter the files by.
        files (List[Path]): A list of file paths to be filtered.

    Returns:
        List[Path]: A list of file paths that contain any of the specified chip IDs in their names.
    """
    filtered_files = []
    for file in files:
        if any(chip_id in file.name for chip_id in chip_ids):
            filtered_files.append(file)
    return filtered_files


def grab_chip_ids(file_path: Path) -> List[str]:
    """
    Reads a CSV file and extracts a list of chip IDs from the 'ids' column.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        List[str]: A list of chip IDs.

    Raises:
        ValueError: If the 'ids' column is not present in the CSV file.
    """
    df = pd.read_csv(file_path)
    if "ids" not in df.columns:
        raise ValueError("The file does not contain the 'ids' column.")
    return list(df["ids"])


def open_tiff(fname: Path) -> np.ndarray:
    """
    Opens a TIFF file and reads its data.

    Parameters:
    fname (Path): The file path to the TIFF file.

    Returns:
    numpy.ndarray: The data read from the TIFF file.
    """
    with rasterio.open(fname, "r") as src:
        data = src.read()
    return data


def plot_distribution(chips_df, title, output_file):
    """
    Plots the distribution of class occurrences in the given DataFrame and saves the plot as a PNG file.

    Args:
        chips_df (pd.DataFrame): DataFrame containing the class data. The DataFrame should have columns corresponding to class indices (0 to 13).
        title (str): Title of the plot.
        output_file (str): Path to the output file where the plot will be saved.

    Returns:
        None
    """
    plt.bar(
        range(14),
        chips_df.loc[:, range(14)].sum() / np.sum(chips_df.loc[:, range(14)].sum()),
    )
    plt.xticks(range(14), list(CROP_CLASS_MAPPING.values()), rotation=90)
    plt.title(title)
    plt.draw()
    plt.savefig(output_file, dpi=150, format="png", bbox_inches="tight")
    plt.close()


def process_filtered_chips(chip_ids: List[str]) -> pd.DataFrame:
    """
    Processes filtered chips and generates a DataFrame with class distributions.
    Args:
        chip_ids (List[str]): A list of chip IDs to filter the chips.
    Returns:
        pd.DataFrame: A DataFrame containing the class distributions for each chip.
                      The DataFrame has columns for chip ID and counts for each class (0-13).
    """
    chips_df = create_empty_df()
    cdl_chips = list(FILTERED_DIR.glob("*mask.tif"))
    filtered_cdl_chips = filter_files_by_chip_ids(chip_ids, cdl_chips)
    for file in filtered_cdl_chips:
        img = open_tiff(file)
        classes, class_counts = np.unique(img, return_counts=True)
        counts = np.zeros(14)
        counts[classes] = class_counts
        chips_df.loc[len(chips_df.index)] = [id] + counts.tolist()

    return chips_df


def main() -> None:
    """
    Main function to generate and plot class distribution for training and validation datasets.
    This function performs the following steps:
    1. Reads chip IDs for validation and training datasets from CSV files.
    2. Processes the filtered chip IDs to create dataframes.
    3. Plots the class distribution for validation and training datasets and saves the plots as PNG files.
    The paths for the CSV files and the output plots are derived from the TRAINING_DATASET_PATH constant.
    """
    val_chips_ids = grab_chip_ids(Path(TRAINING_DATASET_PATH, "val_ids.csv"))
    train_chip_ids = grab_chip_ids(Path(TRAINING_DATASET_PATH, "train_ids.csv"))

    val_chips_df = process_filtered_chips(val_chips_ids)
    plot_distribution(
        val_chips_df,
        "Validation Data Distribution",
        Path(TRAINING_DATASET_PATH, "validation_dst.png"),
    )

    train_chips_df = process_filtered_chips(train_chip_ids)
    plot_distribution(
        train_chips_df,
        "Training Data Distribution",
        Path(TRAINING_DATASET_PATH, "training_dst.png"),
    )


if __name__ == "__main__":
    main()
