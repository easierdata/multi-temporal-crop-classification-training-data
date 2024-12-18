import rasterio
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

from pathlib import Path

import sys

module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


def calc_mean_std(kwargs) -> None:
    """
    Calculate the mean, standard deviation, minimum, and maximum values of a specified band
    from a collection of raster files and save these statistics to text files. Additionally,
    save all band values to a binary file.
    Args:
        kwargs (dict): A dictionary containing the following keys:
            - "i_band" (int): The index of the band.
            - "band" (int): The band number to process.
    Returns:
        None
    """
    i_band = kwargs["i_band"]
    band = kwargs["band"]
    clip_val = 1.5

    merged_hls_chips = list(FILTERED_DIR.glob("*merged.tif"))
    # means = []
    # stds = []

    print("Bands:", band)
    vals_all = np.ndarray([0])
    for k in range(len(merged_hls_chips)):
        file = merged_hls_chips[k]
        with rasterio.open(file) as src:
            vals = src.read(band).flatten()
            vals_all = np.concatenate([vals_all, vals])
    band_mean = vals_all.mean()
    band_std = vals_all.std()
    vals_all_clipped = np.clip(
        vals_all,
        np.nanpercentile(vals_all, clip_val),
        np.nanpercentile(vals_all, 100 - clip_val),
    )
    band_min = np.min(vals_all_clipped)
    band_max = np.max(vals_all_clipped)
    with open(
        Path(TRAINING_DATASET_PATH, "band_stats", f"band_values_{str(i_band)}.txt"),
        "w+",
    ) as f:
        f.write(f"Mean: {band_mean}\n")
        f.write(f"Std: {band_std}\n")
        f.write(f"Min: {band_min}\n")
        f.write(f"Max: {band_max}\n")

    with open(
        Path(TRAINING_DATASET_PATH, "band_stats", f"band_values_{str(i_band)}.list"),
        "wb",
    ) as file:
        pickle.dump(vals_all, file)


def generate_global_stats() -> None:
    # Initialize lists to store the values
    min_values = []
    max_values = []
    mean_values = []
    std_values = []
    band_stats_dir = Path(TRAINING_DATASET_PATH, "band_stats").resolve()
    band_stats_txt_files = list(band_stats_dir.glob("*.txt"))
    band_stats_txt_files.sort()
    # Loop through the 6 text files
    for filename in band_stats_txt_files:
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("Min:"):
                    min_values.append(float(line.split(": ")[1].strip()))
                elif line.startswith("Max:"):
                    max_values.append(float(line.split(": ")[1].strip()))
                elif line.startswith("Mean:"):
                    mean_values.append(float(line.split(": ")[1].strip()))
                elif line.startswith("Std:"):
                    std_values.append(float(line.split(": ")[1].strip()))

    # Print the lists in the desired format
    print(f"min: {min_values}")
    print(f"max: {max_values}")
    print(f"mean: {mean_values}")
    print(f"std: {std_values}")

    # Save each list as a row to a text file in the training dataset directory
    with open(Path(TRAINING_DATASET_PATH, "global_stats.txt"), "w") as file:
        file.write(f"min: {min_values}\n")
        file.write(f"max: {max_values}\n")
        file.write(f"mean: {mean_values}\n")
        file.write(f"std: {std_values}\n")


def main() -> None:
    """
    Main function to calculate mean and standard deviation for each band.
    This function performs the following steps:
    1. Creates a directory named 'band_stats' inside the MISC_DIR directory.
    2. Defines a list of band indices and corresponding band IDs.
    3. Creates a DataFrame with the band indices and band IDs.
    4. Uses multiprocessing to calculate the mean and standard deviation for each band in parallel.
    Note:
        - The function assumes that the `MISC_DIR` variable, `Path` class, `pd` (pandas),
          `mp` (multiprocessing), and `calc_mean_std` function are defined elsewhere in the code.
    """

    # Create band stats directory
    Path(TRAINING_DATASET_PATH, "band_stats").mkdir(parents=True, exist_ok=True)

    i_band = list(range(1, 7))
    band_ids = [
        [1, 7, 13],
        [2, 8, 14],
        [3, 9, 15],
        [4, 10, 16],
        [5, 11, 17],
        [6, 12, 18],
    ]
    df = pd.DataFrame({"i_band": i_band, "band": band_ids})
    with mp.Pool(processes=6) as pool:
        pool.map(calc_mean_std, df.to_dict("records"))

    # Save HLS band stats to a text file
    generate_global_stats()


if __name__ == "__main__":
    main()
