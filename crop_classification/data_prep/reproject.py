import json
import os
from pathlib import Path
import sys
from rasterio.enums import Resampling
import pyproj
import pandas as pd
import numpy as np
import multiprocessing as mp
import rioxarray

# The code cell is used to add the src directory to the Python path, making
# it possible to import modules from that directory.

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

# Constants
TARGET_CRS = "EPSG:5070"
RESAMPLING_METHOD = Resampling.bilinear
PROCESSES = 5
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]


# Functions
def transform_point(coor, src_crs, target_crs=TARGET_CRS):
    # Transforms a point from one coordinate system to another
    proj = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    return proj.transform(coor[0], coor[1])


def get_nearest_value(array, value):
    # Finds the value in the array that is closest to the given value
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def reproject_tile(
    tile_path, cdl_ds, remove_original=False, resampling_method=RESAMPLING_METHOD
):
    """
    This function receives the path to a specific HLS tile and reproject it to the targeting crs_ds.
    The option of removing the raw HLS tile is provided

    Assumptions:
    - tile_path is a full path that end with .tif
    - cdl_ds is a rioxarray dataset that is opened with `cache=False` setting.


    Inputs:
    - tile_path: The full path to a specific HLS tile
    - target_crs: The crs that you wish to reproject the tile to, default is EPSG 4326
    - remove_original: The option to remove raw HLS tile after reprojecting, default is True
    - resampling_method: The method that rioxarray use to reproject, default is bilinear
    """

    xds = rioxarray.open_rasterio(tile_path)
    half_scene_len = np.abs(np.round((xds.x.max().data - xds.x.min().data) / 2))
    coor_min = transform_point(
        [xds.x.min().data - half_scene_len, xds.y.min().data - half_scene_len],
        xds.rio.crs,
    )
    coor_max = transform_point(
        [xds.x.max().data + half_scene_len, xds.y.max().data + half_scene_len],
        xds.rio.crs,
    )

    x0 = get_nearest_value(cdl_ds.x.data, coor_min[0])
    y0 = get_nearest_value(cdl_ds.y.data, coor_min[1])
    x1 = get_nearest_value(cdl_ds.x.data, coor_max[0])
    y1 = get_nearest_value(cdl_ds.y.data, coor_max[1])

    cdl_for_reprojection = cdl_ds.rio.slice_xy(x0, y0, x1, y1)

    xds_new = xds.rio.reproject_match(
        cdl_for_reprojection, resampling=resampling_method
    )

    if remove_original:
        if Path(tile_path).is_file():
            os.remove(tile_path)
        xds_new.rio.to_raster(raster_path=tile_path.replace(".tif", ".reproject.tif"))
    else:
        xds_new.rio.to_raster(raster_path=tile_path.replace(".tif", ".reproject.tif"))


def process_tile(kwargs):
    remove_original = True

    save_path = kwargs["save_path"]
    filename = kwargs["filename"]
    bands = json.loads(kwargs["bands"])
    cdl_file = kwargs["cdl_file"]

    cdl_ds = rioxarray.open_rasterio(cdl_file, cache=False)

    for band in bands:
        tile_path = f"{save_path}{filename}.{band}.tif"
        if Path(tile_path).is_file():
            if band == "Fmask":
                reproject_tile(
                    tile_path,
                    cdl_ds,
                    remove_original,
                    resampling_method=Resampling.nearest,
                )
            else:
                reproject_tile(tile_path, cdl_ds, remove_original)


def main():
    # Loads the track_df file and adds the cdl_file and bands columns
    # TODO - Add argparse param for passing in the path to the dataframe object
    track_df = pd.read_pickle(TILES_DF_PKL)

    # TODO - Add argparse param for passing in the path to the cdl file. If it is not passed in, use the default path should
    # point to the CDL file in the data folder.
    track_df["cdl_file"] = CDL_SOURCE

    track_df["bands"] = json.dumps(BANDS)

    # Reprojects HLS tiles to the same CRS as CDL
    with mp.Pool(processes=PROCESSES) as pool:
        pool.map(process_tile, track_df.to_dict("records"))


if __name__ == "__main__":
    main()