import pandas as pd
import os
import json

tiles = pd.read_pickle(
    r"C:\github\client_projects\umd\multi-temporal-crop-classification-training-data\data\selected_tiles_df.pkl"
)
# Load the JSON data
with open(
    r"C:\github\client_projects\umd\multi-temporal-crop-classification-training-data\data\misc\hls-imagery-cid-output.json",
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

# Convert the dictionary to a DataFrame and export it to a JSON file
df = pd.DataFrame(final_mapping.items(), columns=["filename", "cid"])
df.to_json(
    r"C:\github\client_projects\umd\multi-temporal-crop-classification-training-data\data\misc\filename_to_cid_mapping.json",
    orient="records",
)
