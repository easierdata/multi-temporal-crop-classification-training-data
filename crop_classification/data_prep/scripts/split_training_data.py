import sys
import pandas as pd
import numpy as np
from pathlib import Path

# used to add the src directory to the Python path, making
# it possible to import modules from that directory.
module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


def main() -> None:
    """
    Main function to split chip IDs into training and validation sets.
    This function performs the following steps:
    1. Retrieves a list of all file paths in the `chips_filtered` directory with the suffix `mask.tif`.
    2. Extracts chip IDs from the filenames by parsing the value between `chip_` and `.mask.tif`.
    3. Randomly splits the list of chip IDs into 80% training and 20% validation sets.
    4. Prints the total number of chips, number of training chips, and number of validation chips.
    5. Saves the training and validation chip IDs to CSV files named `train_ids.csv` and `val_ids.csv` to the training dataset
       specified in the configuration file.
    """
    # get a list of all the filepaths in the `chips_filtered` directory
    filepaths = list(FILTERED_DIR.glob("*mask.tif"))
    filenames = [file.name for file in filepaths]

    # Extract the value between the first `chip_` and `.mask.tif` to get the chip id
    chip_ids = [file.split("chip_")[1].split(".mask.tif")[0] for file in filenames]

    # Randomly split the list of chip ids into 80% training and 20% validation
    np.random.seed(42)
    np.random.shuffle(chip_ids)
    split = int(0.8 * len(chip_ids))
    train_ids = chip_ids[:split]
    val_ids = chip_ids[split:]

    print(f"Total number of chips generated: {len(chip_ids)}")
    print(f"Number of training chips: {len(train_ids)}")
    print(f"Number of validation chips: {len(val_ids)}")

    # Save the training and validation chip ids to csv file
    train_ids_df = pd.DataFrame(train_ids, columns=["ids"])
    val_ids_df = pd.DataFrame(val_ids, columns=["ids"])
    train_ids_df.to_csv(Path(TRAINING_DATASET_PATH, "train_ids.csv"), index=False)
    val_ids_df.to_csv(Path(TRAINING_DATASET_PATH, "val_ids.csv"), index=False)


if __name__ == "__main__":
    main()
