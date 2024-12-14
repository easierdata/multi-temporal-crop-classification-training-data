from pathlib import Path
import pandas as pd
from ipfs_stac import client
import requests
import tqdm
import sys
import json

module_path = Path(__file__).parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

# Constants

# Pull out all parameters from `IPFS_STAC` that are not empty or None and create a client object
params = {k: v for k, v in IPFS_STAC.items() if v}
easier = client.Web3(**params)


def process_tile(easier: client.Web3, tile_info: dict) -> None:

    title_id = tile_info["title_id"]
    scene_dir = Path(TILE_DIR, title_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing tile {title_id}")

    try:
        print("Searching STAC...")
        item = easier.searchSTAC(ids=[title_id])
        # If the item is not found, skip to the next tile
        if not item:
            print(
                f"The item, {title_id}, does not exist in the STAC catalog. Skipping..."
            )
            return

        for band in HLS_BANDS:
            # print(f"Processing band {band}")
            try:
                asset = easier.getAssetFromItem(item[0], band)
                # print(f"Asset retrieved successfully {band}")
                cid = str(asset)[:59]
                asset.cid = cid
                # print(f"CID retrieved successfully: {cid}")
                output_path = scene_dir / f"{title_id}.{band}.tif"

                # create a dictionary to CID payload

                if not output_path.exists():
                    print(f"Attempting to get data for CID: {cid}")
                    try:
                        if asset.fetch():
                            # data = easier.getFromCID(cid)
                            # print(f"Fetch successful for {band}")

                            print(f"Data retrieved successfully for {band}")
                            with open(output_path, "wb") as f:
                                f.write(asset.data)
                            print(f"Downloaded {band} for {title_id}")
                    except Exception as e:
                        print(f"Retrieval error for {band}: {str(e)}")
                # else:
                #     print(f"Skipping {band} for {title_id} - already exists")
            except Exception as e:
                print(f"Asset error for {band}: {str(e)}")
    except Exception as e:
        print(f"Tile error: {str(e)}")


def main():
    """Main function to coordinate the download process."""
    # Read the selected tiles
    try:
        tiles_df = pd.read_csv(SELECTED_TILES_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {SELECTED_TILES_CSV}")
        return

    print(f"Found {len(tiles_df)} tiles to download")

    # Process each tile
    for _, tile_info in tqdm.tqdm(tiles_df.iterrows(), total=len(tiles_df)):
        process_tile(easier, tile_info)


if __name__ == "__main__":
    main()
