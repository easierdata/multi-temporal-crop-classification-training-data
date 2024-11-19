# Loading chips bounding boxes from geojson
import datetime
import json
import os
import sys

import fiona
import numpy as np
import pandas as pd
from tqdm import tqdm
import xmltodict
import geopandas
from pystac_client import Client
from pathlib import Path
import earthaccess

earthaccess.login(strategy="netrc")

# The code cell is used to add the src directory to the Python path, making
# it possible to import modules from that directory.

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

# STATIC VARIABLES
CLOUD_THRES = 5  # percent cloud cover for tile level query
SELECTION_SUBSET = None  # number of chips to select


def find_tile(x, y):
    """
    Identify closest tile
    """

    s = (tile_x - x) ** 2 + (tile_y - y) ** 2
    tname = tile_name[np.argmin(s)]
    return tname


### Filter based on spatial coverage
def spatial_filtering(dataframe):
    """
    Using spatial coverage percentage to filter chips

    Args:
        dataframe: A pandas dataframe that generated previously
    """
    cover_list = [100, 90, 80, 70, 60, 50]
    tile_list_ft = []
    tile_list = dataframe.tile_id.unique().tolist()

    for tile in tqdm(tile_list):
        temp_df = dataframe[dataframe.tile_id == tile]
        for cover_pct in cover_list:

            temp_df_filtered = temp_df[temp_df.spatial_cover >= cover_pct]
            if len(temp_df_filtered) >= 3:
                for i in range(len(temp_df_filtered)):
                    tile_list_ft.append(temp_df_filtered.iloc[i])
                break

    tile_df_filtered = pd.DataFrame(tile_list_ft)
    return tile_df_filtered


def select_scenes(dataframe):
    """
    Selecting best spatial covered scenes based on timesteps

    Args:
        dataframe: A pandas dataframe that generated previously
    """
    select_tiles = []
    tile_list = dataframe.tile_id.unique().tolist()

    for tile in tqdm(tile_list):
        temp_df = (
            dataframe[dataframe.tile_id == tile]
            .sort_values("date")
            .reset_index(drop=True)
        )
        select_tiles.extend(
            [temp_df.iloc[0], temp_df.iloc[len(temp_df) // 2], temp_df.iloc[-1]]
        )

    return pd.DataFrame(select_tiles).reset_index(drop=True)


# Open bounding box chip payload. We'll need the chips variable later as it contains the geometric info
with open(BB_CHIP_PAYLOAD, "r") as file:
    chips = json.load(file)

# Filter list of chips to just a handful
if SELECTION_SUBSET:
    chips_subset = chips["features"][:SELECTION_SUBSET]
    chips["features"] = chips_subset

# Create lists about chip information to find tiles corresponding to it later
chip_ids = []
chip_x = []
chip_y = []
for item in chips["features"]:
    chip_ids.append(item["properties"]["id"])
    chip_x.append(item["properties"]["center"][0])
    chip_y.append(item["properties"]["center"][1])

with open(CHIPS_ID_JSON, "w") as f:
    json.dump(chip_ids, f, indent=2)

# does CHIPS_DF_PKL exist? if not, create the required dataframe and then export it to a pickle file
if not os.path.exists(CHIPS_DF_PKL):

    # Read in sentinel kml file
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    tile_src = geopandas.read_file(HLS_KML_FILE, driver="KML")

    # Convert the CRS to a projected one (Web Mercator)
    # tile_src = tile_src.to_crs("EPSG:3857")

    # # Create table containing information about sentinel tiles
    # directly assign the values to the new lists using pandas operations which are optimized for performance
    tile_name = tile_src["Name"].values
    tile_x = tile_src["geometry"].centroid.x.values
    tile_y = tile_src["geometry"].centroid.y.values
    tile_src = pd.concat([tile_src, tile_src.bounds], axis=1)

    # Add tile information to the chip dataframe
    chip_df = pd.DataFrame({"chip_id": chip_ids, "chip_x": chip_x, "chip_y": chip_y})
    chip_df["tile"] = chip_df.apply(
        lambda row: find_tile(row["chip_x"], row["chip_y"]), axis=1
    )

    # Save dataframe to csv for later uses
    chip_df.to_pickle(CHIPS_DF_PKL)
else:
    chip_df = pd.read_pickle(CHIPS_DF_PKL)

# Get unique tiles of tiles from the chip dataframe
tiles = chip_df.tile.unique().tolist()

archive = False
if not os.path.exists(TILES_DF_PKL):
    ### Querying tile links based on geometry of chips
    STAC_URL = "https://cmr.earthdata.nasa.gov/stac"
    catalog = Client.open(f"{STAC_URL}/LPCLOUD/")
    tile_list = []
    print(f"There are a total of {len(tiles)} tiles")
    tile_iter = 0
    for current_tile in tiles:

        # Get the first chip of the tile to get the geometry
        chip_df_filt = chip_df.loc[chip_df.tile == current_tile]
        first_chip_id = chip_df_filt.chip_id.iloc[0]
        chip_feature = [
            feature
            for feature in chips["features"]
            if feature.get("properties", {}).get("id") == first_chip_id
        ]

        roi = chip_feature[0]["geometry"]

        search = catalog.search(
            collections=["HLSS30.v2.0"],
            intersects=roi,
            datetime="2022-03-01/2022-09-30",
        )

        # Create earthaccess session
        fs = earthaccess.get_fsspec_https_session()

        num_results = search.matched()
        item_collection = search.item_collection()

        tile_name = "T" + current_tile
        iter_items = 0
        for i in tqdm(item_collection, desc=f"({tile_iter}/{len(tiles)})"):
            if i.id.split(".")[2] == tile_name:
                # Tile-level 5% cloud cover threshold query
                if i.properties["eo:cloud_cover"] <= CLOUD_THRES:
                    with fs.open(i.assets["metadata"].href) as xml:
                        if xml.r.status == 200:
                            temp_xml = xml.read()
                            temp_xml = xmltodict.parse(temp_xml)
                            temp_dict = {
                                "tile_id": tile_name,
                                "cloud_cover": i.properties["eo:cloud_cover"],
                                "date": datetime.datetime.strptime(
                                    i.properties["datetime"].split("T")[0], "%Y-%m-%d"
                                ),
                                "spatial_cover": int(
                                    temp_xml["Granule"]["AdditionalAttributes"][
                                        "AdditionalAttribute"
                                    ][3]["Values"]["Value"]
                                ),
                                "http_links": {
                                    "B02": i.assets["B02"].href,
                                    "B03": i.assets["B03"].href,
                                    "B04": i.assets["B04"].href,
                                    "B8A": i.assets["B8A"].href,
                                    "B11": i.assets["B11"].href,
                                    "B12": i.assets["B12"].href,
                                    "Fmask": i.assets["Fmask"],
                                },
                                "s3_links": {
                                    "B02": i.assets["B02"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "B03": i.assets["B03"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "B04": i.assets["B04"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "B8A": i.assets["B8A"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "B11": i.assets["B11"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "B12": i.assets["B12"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                    "Fmask": i.assets["Fmask"].href.replace(
                                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/",
                                        "s3:/",
                                    ),
                                },
                            }
                            tile_list.append(temp_dict)
                            iter_items += 1
                        else:
                            assert (
                                False
                            ), f"Failed to fetch XML from {i.assets['metadata'].href}. Error code: {xml.r.status}"

        tile_iter += 1

    tile_df = pd.DataFrame(tile_list)

    # Save to csv for later uses
    tile_df.to_pickle(TILES_DF_PKL)
    tile_df = pd.read_pickle(TILES_DF_PKL)

    cover_df = spatial_filtering(tile_df)

    selected_tiles = select_scenes(cover_df)
    # Save for later uses
    selected_tiles.to_csv(SELECTED_TILES_CSV, index=False)
