from typing import Dict, Union
import pandas as pd
import json
import tqdm
import subprocess
from pathlib import Path
import sys
from ipfs_stac import client

module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


# Pull out all parameters from `IPFS_STAC` that are not empty or None and create a client object
params = {k: v for k, v in IPFS_STAC.items() if v}
# easier = client.Web3(**params)


def retrieve_assets(data_payload: Dict) -> None:
    """
    Retrieve assets from IPFS based on the provided data payload.
    This function iterates over the keys in the data payload, which represent different scenes.
    For each scene, it creates a directory (if it does not already exist) to store the assets.
    It then iterates over the assets in the scene, retrieves each asset using its CID, and stores it in the corresponding directory.
    Args:
        data_payload (Dict): A dictionary where keys are scene names and values are lists of assets.
                             Each asset is represented as a dictionary with at least the following keys:
                             - "cid": The content identifier (CID) for the asset in IPFS.
                             - "filename": The name of the file to be saved locally.
    Returns:
        None
    """
    for scene in tqdm.tqdm(data_payload.keys()):
        scene_data = data_payload[scene]
        # Create a directory to store content by scene name if it does not exist
        scene_dir = Path(TILE_DIR, scene)
        scene_dir.mkdir(parents=True, exist_ok=True)
        for asset in scene_data:
            cid = asset["cid"]
            # Create filepath to store the downloaded asset
            output_path = Path(scene_dir, asset["filename"])
            if not output_path.exists():
                # easier.writeCID(cid, output_path)

                # Run subprocess call to download content using ipfs get -o command
                result = subprocess.run(
                    ["ipfs", "get", cid, "-o", output_path, "-p"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result.returncode != 0:
                    print(f"Error downloading {cid}")
                    print(result.stderr.decode("utf-8"))
                else:
                    print(result.stdout.decode("utf-8"))


def main() -> None:
    """
    Main function to read selected tiles, create payloads, and retrieve assets.

    This function performs the following steps:
    1. Reads tile payloads to a JSON file.
    2. Retrieves the assets for each tile using the payload information.

    If the tile payloads JSON file is not found, an error message is printed and the function returns.
    """
    # Read in the output from running the `grab_cids_from_selected_tiles.py` script
    tile_payloads_file = Path(TRAINING_DATASET_PATH, "tile_payloads.json").resolve()
    if not tile_payloads_file.exists():
        raise FileNotFoundError(f"Could not find {tile_payloads_file}")

    # open json file
    with open(tile_payloads_file, "r") as file:
        all_tile_payloads = json.load(file)

    # Retrieve the assets for each tile
    retrieve_assets(all_tile_payloads)


if __name__ == "__main__":
    main()
