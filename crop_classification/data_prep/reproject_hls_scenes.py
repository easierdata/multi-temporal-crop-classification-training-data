from pathlib import Path
import sys
from typing import Tuple, Union, Dict, Any
from rasterio.enums import Resampling
import pyproj
import pandas as pd
import numpy as np
import rioxarray
from tqdm.contrib.concurrent import process_map

# used to add the src directory to the Python path, making
# it possible to import modules from that directory.
module_path = Path(__file__).parent.parent.resolve().as_posix()
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
OVERWRITE_EXISTING = False

# Global variable for the CDL dataset
CDL_DS = rioxarray.open_rasterio(CDL_SOURCE, cache=True, driver="GTiff", compress="lzw")

# CDL_DS = None


def transform_point(
    coor: Tuple, src_crs: str, target_crs: str = TARGET_CRS
) -> Tuple[Any, Any]:
    """
    Transforms a point from one coordinate system to another.

    Args:
        coor (tuple): The coordinates of the point to be transformed.
        src_crs (str): The source coordinate reference system (CRS) of the point.
        target_crs (str, optional): The target CRS to transform the point to. Defaults to TARGET_CRS.

    Returns:
        tuple: The transformed coordinates of the point.
    """
    proj = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    return proj.transform(coor[0], coor[1])


def get_nearest_value(array: np.ndarray, value: float) -> float:
    """
    Finds the value in the array that is closest to the given value.

    Parameters:
    array (numpy.ndarray): The input array.
    value (float): The value to find the closest match for.

    Returns:
    float: The value in the array that is closest to the given value.
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def reproject_tile(
    tile_path: Union[str, Path],
    resampling_method: Resampling = RESAMPLING_METHOD,
) -> None:
    """
    This function receives the path to a specific HLS tile and reproject it to the coordinate system defined with `TARGET_CRS`.

    Assumptions:
    - tile_path is a full path that end with .tif
    - CDL_DS is a rioxarray dataset that is opened and will be cached in memory


    Inputs:
    - tile_path: The full path to a specific HLS tile
    - resampling_method: The method that rioxarray use to reproject, default is bilinear
    """

    # cdl_ds = rioxarray.open_rasterio(CDL_SOURCE, driver="GTiff", compress="lzw")
    # global CDL_DS
    # Open the tile and reproject it to the same CRS as the CDL
    # Handling file operations using the `with` statement as a context manager
    with rioxarray.open_rasterio(tile_path, driver="GTiff", compress="lzw") as xds:
        half_scene_len = np.abs(np.round((xds.x.max().data - xds.x.min().data) / 2))
        coor_min = transform_point(
            [xds.x.min().data - half_scene_len, xds.y.min().data - half_scene_len],
            xds.rio.crs,
        )
        coor_max = transform_point(
            [xds.x.max().data + half_scene_len, xds.y.max().data + half_scene_len],
            xds.rio.crs,
        )

        x0 = get_nearest_value(CDL_DS.x.data, coor_min[0])
        y0 = get_nearest_value(CDL_DS.y.data, coor_min[1])
        x1 = get_nearest_value(CDL_DS.x.data, coor_max[0])
        y1 = get_nearest_value(CDL_DS.y.data, coor_max[1])

        cdl_for_reprojection = CDL_DS.rio.slice_xy(x0, y0, x1, y1)

        xds_new = xds.rio.reproject_match(
            cdl_for_reprojection, resampling=resampling_method
        )

    # Save the reprojected tile to reprojected directory as not to interfere with the original tiles
    reprojected_tile_path = Path(TILE_REPROJECTED_DIR, tile_path.name)
    xds_new.rio.to_raster(reprojected_tile_path, driver="GTiff", compress="lzw")


def process_tile(tile_payload: Dict) -> None:
    """
    Process a tile by reprojecting the bands.

    Args:
        tile_payload (dict): A dictionary containing information about the tile.

    Returns:
        None
    """

    # Extract the filename from the dictionary as to populate the filename with the band ID
    filename = tile_payload["title_id"]

    # check if tile directory is empty
    if not list(Path(TILE_DIR).rglob(f"{filename}*.tif")):
        print(f"No files for {filename}. Skipping...")
        return

    for band in HLS_BANDS:
        tile_path = Path(TILE_DIR) / f"{filename}/{filename}.{band}.tif"
        reprojected_file_path = Path(TILE_REPROJECTED_DIR, f"{filename}.{band}.tif")
        if not OVERWRITE_EXISTING:
            if reprojected_file_path.exists():
                # print(
                #     f"Reprojected file {reprojected_file_path} already exists. Skipping..."
                # )
                continue

        if tile_payload["remove_original"]:
            if Path(tile_path).is_file():
                Path.unlink(tile_path)

        if Path(tile_path).is_file():
            try:
                if band == "Fmask":
                    reproject_tile(
                        tile_path=tile_path, resampling_method=Resampling.nearest
                    )
                else:
                    reproject_tile(tile_path=tile_path)
            except Exception as e:
                print(f"Failed to reproject {tile_path}: {e}")

        elif not Path(tile_path).exists() and not tile_payload["remove_original"]:
            print(f"Warning: {tile_path.name} does not exist")
            continue


def main() -> None:
    # Load in the dataframe containing the selected tiles identified in the `identify_hls_scenes.py` script
    track_df = pd.read_pickle(SELECTED_TILES_PKL)

    # Let's ensure the geojson used to coordinate transformation (EPSG:5070) has been created.
    # The following script can be manually ran `create_chip_5070_payload.py` script but we run the
    # script here to ensure that the file is available for reprojecting the imagery.
    if not Path(BB_CHIP_5070_PAYLOAD).exists():
        from data_prep.scripts.create_chip_5070_payload import (
            main as create_chip_5070_payload_main,
        )

        create_chip_5070_payload_main()

    remove_original = False
    track_df["remove_original"] = remove_original
    # print(track_df.head(), track_df.shape)
    process_map(
        process_tile,
        track_df.to_dict("records"),
        max_workers=PROCESSES,
        desc="Processing tiles",
    )


if __name__ == "__main__":
    main()
