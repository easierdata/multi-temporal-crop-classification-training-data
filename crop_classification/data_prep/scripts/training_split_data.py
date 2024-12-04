import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# used to add the src directory to the Python path, making
# it possible to import modules from that directory.
module_path = module_path = os.path.abspath(
    Path(__file__).parent.parent.parent.resolve()
)
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


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
train_ids_df.to_csv(Path(DATA_DIR, "train_ids.csv"), index=False)
val_ids_df.to_csv(Path(DATA_DIR, "val_ids.csv"), index=False)
