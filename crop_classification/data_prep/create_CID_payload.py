import pandas as pd
import os
import json
from pathlib import Path
import sys

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

tiles = pd.read_pickle(
    SELECTED_TILES_PKL
)
# Load the JSON data
with open(
    MISC_DIR / "hls-imagery-cid-output.json",
    "r",
) as file:
    data = json.load(file)

# Initialize an empty list to store the dictionaries of filenames
filenames_list = []

# Iterate over each row in the DataFrame
for index, row in tiles.iterrows():
    # Access the dictionary in the `http_links` column
    links_dict = row["http_links"]

    # Iterate over the dictionary of links
    for key, url in links_dict.items():
        # Extract the filename from the URL
        filenames_list.append(os.path.basename(url))

# Create an empty dictionary for filename to CID mapping
filename_to_cid = {}

# Iterate through the subEntries to map filename to CID
for entry in data["subEntries"]:
    path = entry["path"]
    cid = entry["cid"]
    filename_to_cid[path] = cid

# Initialize an empty dictionary for the final mapping
final_mapping = {}

# Loop through each filename in the list
for item in filenames_list:
    # Check if the filename exists in the filename_to_cid mapping
    if item in filename_to_cid:
        # Add the filename and its corresponding CID value to the final mapping
        final_mapping[item] = filename_to_cid[item]

# final_mapping now contains your desired mapping
print(final_mapping)

# export the dictionary to a JSON file
with open(
    MISC_DIR / "selected_cid_mapping.json",
    "w",
) as file:
    json.dump(final_mapping, file)