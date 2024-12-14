import rasterio
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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


def calculate_class_weights(files: Path) -> List[np.float64]:
    """
    Calculate class weights for a set of TIFF files.

    This function reads multiple TIFF files, flattens their data, and computes
    class weights based on the frequency of each class. The weights are normalized
    so that their sum equals 1.

    Args:
        files (Path): A list of file paths to the TIFF files.

    Returns:
        np.ndarray: An array of normalized class weights.
    """
    y = []
    for file in files:
        data = open_tiff(file)
        y.append(data.flatten())
    y_stack = np.vstack(y)
    y_flatten = y_stack.flatten()
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(np.ravel(y_flatten, order="C")),
        y=np.ravel(y_flatten, order="C"),
    )
    return list(class_weights / np.sum(class_weights))


def main() -> None:
    """
    Main function to calculate and save class weights for crop classification.

    This function performs the following steps:
    1. Retrieves a list of filtered mask files from the specified directory.
    2. Raises a FileNotFoundError if no mask files are found.
    3. Calculates class weights based on the retrieved mask files.
    4. Inserts a weight of 0 for the background class (no crop).
    5. Saves the calculated class weights to a CSV file in the specified training dataset path.

    Raises:
        FileNotFoundError: If no mask files are found in the specified directory.
    """
    files = list(FILTERED_DIR.glob("*mask.tif"))
    if not files:
        raise FileNotFoundError(
            f"No files found in {FILTERED_DIR.resolve().as_posix()}."
        )
    class_weights = calculate_class_weights(files)
    # Insert 0 weight for the background class (no crop = 0)
    class_weights.insert(0, np.float64(0))
    # Save class weights to file
    with open(Path(TRAINING_DATASET_PATH, "class_weights.csv"), "w") as output_file:
        for idx, weight in enumerate(class_weights):
            output_file.write(f"{idx}, {weight}\n")


if __name__ == "__main__":
    main()
