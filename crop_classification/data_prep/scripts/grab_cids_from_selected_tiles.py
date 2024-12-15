from typing import Dict, Union
import pandas as pd
import json
import tqdm
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
easier = client.Web3(**params)


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
        item = easier.searchSTAC(ids=[title_id])
        # If the item is not found, return an empty dictionary
        if not item:
            print(f"\nThe item, {title_id}, does not exist in the STAC catalog.")
            return None

        for band in HLS_BANDS:
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


def main() -> None:
    """
    Main function to read the selected tiles file and generate a json containing the CIDs for each tile.
    This function performs the following steps:
    1. Reads the selected tiles from a CSV file.
    2. Creates a JSON file to store tile payload information for each tile.
    3. Processes each tile to create its payload and stores it in a dictionary.
    4. Saves the tile payloads to a JSON file.
    If the selected tiles CSV file is not found, an error message is printed and the function returns.

    Raises:
        FileNotFoundError: If the input file "tile_payloads.json" cannot be found or opened.

    """
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
    tile_payloads_file = (
        Path(TRAINING_DATASET_PATH, "tile_payloads.json").resolve().as_posix()
    )
    with open(tile_payloads_file, "w") as f:
        json.dump(all_tile_payloads, f)


if __name__ == "__main__":
    main()
