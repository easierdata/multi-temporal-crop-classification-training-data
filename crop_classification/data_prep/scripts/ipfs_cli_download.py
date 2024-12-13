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

# Constants
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]


# Pull out all parameters from `IPFS_STAC` that are not empty or None and create a client object
params = {k: v for k, v in IPFS_STAC.items() if v}
easier = client.Web3(**params)


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
                print(result.stdout)


def create_tile_payload(easier: client.Web3, tile_info: Dict) -> Union[Dict, None]:
    """
    Creates a payload for a tile by searching for the tile in the STAC catalog and retrieving assets for specified bands.
    Args:
        easier (client.Web3): An instance of the Web3 client used to interact with the STAC catalog.
        tile_info (Dict): A dictionary containing information about the tile, including the "title_id".
    Returns:
        Union[Dict, None]: A list of dictionaries, each containing information about a band asset, or None if the tile is not found.
            Each dictionary in the list contains:
                - "filename" (str): The filename of the band asset.
                - "cid" (str): The content identifier (CID) of the band asset.
                - "band_name" (str): The name of the band.
    """

    title_id = tile_info["title_id"]
    tile_payload = []

    try:
        print("Searching STAC...")
        item = easier.searchSTAC(ids=[title_id])
        # If the item is not found, return an empty dictionary
        if not item:
            print(f"The item, {title_id}, does not exist in the STAC catalog.")
            return None

        for band in BANDS:
            try:
                asset = easier.getAssetFromItem(item[0], band)
                cid = str(asset)[:59]
                asset.cid = cid
                band_payload = {
                    "filename": f"{title_id}.{band}.tif",
                    "cid": asset.cid,
                    "band_name": band,
                }
                tile_payload.append(band_payload)
            except Exception as e:
                print(f"Asset error for {band}: {str(e)}")
    except Exception as e:
        print(f"Tile error: {str(e)}")

    return tile_payload


def main():
    """Main function to coordinate the download process."""

    # Read the selected tiles
    try:
        tiles_df = pd.read_csv(SELECTED_TILES_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {SELECTED_TILES_CSV}")
        return

    # Create a json file to store tile payload information for each tile
    all_tile_payloads = {}
    # Process each tile
    for _, tile_info in tqdm.tqdm(tiles_df.iterrows(), total=len(tiles_df)):
        tile_payload = create_tile_payload(easier, tile_info)
        if tile_payload:
            all_tile_payloads[tile_info.title_id] = tile_payload

    # # Save the tile payloads to a JSON file
    tile_payloads_file = Path(MISC_DIR, "tile_payloads.json").resolve().as_posix()
    with open(tile_payloads_file, "w") as f:
        json.dump(all_tile_payloads, f)
    try:
        if not all_tile_payloads:
            with open(
                tile_payloads_file,
                "r",
            ) as file:
                all_tile_payloads = json.load(file)
        else:
            raise FileNotFoundError()

        # Retrieve the assets for each tile
        retrieve_assets(all_tile_payloads)
    except FileNotFoundError as e:
        print(f"Error: Unable to continue - {str(e)}")
        return


if __name__ == "__main__":
    main()
