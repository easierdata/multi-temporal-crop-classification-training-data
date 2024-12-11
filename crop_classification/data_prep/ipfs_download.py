from pathlib import Path
import pandas as pd
from ipfs_stac import client
import requests
import tqdm

# Get paths relative to script location
SCRIPT_DIR = Path(__file__).parent.parent.parent  # crop_classification folder
DATA_DIR = SCRIPT_DIR / "data"

# Path to input CSV
CSV_PATH = DATA_DIR / "selected_tiles.csv"

# Path where tiles will be downloaded
DOWNLOAD_DIR = DATA_DIR / "download" / "training" / "data" / "tiles"

# Constants
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]

easier = client.Web3(
    # local_gateway="localhost",
    # gateway_port=8080,
    # api_port=5001,
    stac_endpoint="https://stac.easierdata.info"
)


def setup_download_directory():
    """Create the nested directory structure for downloads."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def setup_client():
    """Create and return the IPFS STAC client."""
    easier = client.Web3(
        # local_gateway="localhost",
        # gateway_port=8080,
        # api_port=5001,
        stac_endpoint="https://stac.easierdata.info"
    )

    try:
        response = requests.post("http://localhost:5001/api/v0/id")
        print(f"IPFS connection test: {response.status_code}")
    except Exception as e:
        print(f"IPFS connection failed: {e}")

    return easier


def process_tile(easier: client.Web3, tile_info: dict) -> None:

    title_id = tile_info["title_id"]
    tile_dir = DOWNLOAD_DIR / title_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing tile {title_id}")

    try:
        print("Searching STAC...")
        item = easier.searchSTAC(ids=[title_id])

        for band in BANDS:
            print(f"Processing band {band}")
            try:
                asset = easier.getAssetFromItem(item[0], band)
                print(f"Asset retrieved successfully {band}")
                cid = str(asset)[:59]
                asset.cid = cid
                print(f"CID retrieved successfully: {cid}")
                output_path = tile_dir / f"{title_id}.{band}.tif"

                if not output_path.exists():
                    print(f"Attempting to get data for CID: {cid}")
                    try:
                        # asset.fetch()
                        data = easier.getFromCID(cid)
                        # print(f"Fetch successful for {band}")
                        print(f"Data retrieved successfully for {band}")
                        with open(output_path, "wb") as f:
                            f.write(data)
                        print(f"Downloaded {band} for {title_id}")
                    except Exception as e:
                        print(f"Retrieval error for {band}: {str(e)}")
                else:
                    print(f"Skipping {band} for {title_id} - already exists")
            except Exception as e:
                print(f"Asset error for {band}: {str(e)}")
    except Exception as e:
        print(f"Tile error: {str(e)}")


def main():
    """Main function to coordinate the download process."""
    # Set up download directory
    setup_download_directory()

    # Initialize IPFS STAC client
    # easier = setup_client()

    # Read the selected tiles
    try:
        tiles_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}")
        return

    print(f"Found {len(tiles_df)} tiles to download")

    # Process each tile
    for _, tile_info in tqdm.tqdm(tiles_df.iterrows(), total=len(tiles_df)):
        process_tile(easier, tile_info)


if __name__ == "__main__":
    main()